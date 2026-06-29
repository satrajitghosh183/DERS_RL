// omni/tools/omni_reward.cpp
//
// Debugger-in-the-loop reward CLI — the OmniTrace debugger exposed as a reward signal for
// shader-synthesis RL. Pure C++; links libomni_core.
//
// Pipeline (each stage is a real measurement from the debugger, not a heuristic):
//   1. compile   GLSL -> SPIR-V via glslangValidator (the compile oracle).
//   2. lift      SPIR-V -> UIR via the hand-written frontend (structural validity).
//   3. execute   run the UIR on the CPU SIMT reference over a grid of fragment
//                coordinates; flag NaN/Inf and degenerate (constant) output.
//   4. score     omni::reward::score() composes the decomposed reward.
//
// The point: a shader can *compile* and still be broken (NaN, all-black, constant). Only
// running it catches that — which is exactly what compile@k misses and the debugger provides.
//
// Usage:
//   omni_reward < shader.frag                 # read GLSL from stdin
//   omni_reward --file shader.frag            # or from a file
//   omni_reward --grid 12 --file shader.frag  # NxN execution probe (default 8)
// Output: one line of JSON to stdout. Exit code 0 always (the JSON carries the verdict).

#include "omni/synth/validator.hpp"
#include "omni/frontends/spirv.hpp"
#include "omni/cpuref/interp.hpp"
#include "omni/uir/ir.hpp"
#include "omni/reward/oracle.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>  // close (mkstemps owns the descriptor)

#if defined(OMNI_HAVE_VULKAN)
#include "omni/gpu/vulkan_capture.hpp"   // real GPU render term — the signal a compiler can't give
#endif

namespace {

using namespace omni;

// ---- RAII temp file: created in the system temp dir, unlinked on scope exit. -------------
class TempFile {
public:
    explicit TempFile(std::string_view suffix) {
        // mkstemps creates the file atomically and keeps the suffix (so glslangValidator
        // infers the shader stage from the .frag / .spv extension).
        std::string tmpl = "/tmp/omni_reward_XXXXXX" + std::string(suffix);
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        const int fd = ::mkstemps(buf.data(), static_cast<int>(suffix.size()));
        if (fd != -1) { ::close(fd); path_.assign(buf.data()); }
        else          { path_ = std::move(tmpl); }  // fall back to a fixed name
    }
    ~TempFile() { std::remove(path_.c_str()); }

    TempFile(const TempFile&) = delete;             // Rule of Five: this owns a filesystem
    TempFile& operator=(const TempFile&) = delete;  // resource, so forbid copies/moves.
    TempFile(TempFile&&) = delete;
    TempFile& operator=(TempFile&&) = delete;

    [[nodiscard]] const std::string& path() const noexcept { return path_; }

    [[nodiscard]] bool write(std::string_view data) const {
        std::ofstream os(path_, std::ios::binary | std::ios::trunc);
        os.write(data.data(), static_cast<std::streamsize>(data.size()));
        return static_cast<bool>(os);
    }

private:
    std::string path_;
};

[[nodiscard]] std::string read_stream(std::istream& in) {
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[nodiscard]] std::string json_escape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += c;
        }
    }
    return out;
}

// glslangValidator: GLSL -> .spv. Returns the spv path written into `spv`, or nullopt.
[[nodiscard]] bool emit_spirv(const std::string& glslang_tool, const std::string& frag_path,
                              const std::string& spv_path) {
    // -V = Vulkan SPIR-V, -g = keep debug names (so the interpreter can resolve globals).
    const std::string cmd = glslang_tool + " -V -g \"" + frag_path + "\" -o \"" + spv_path +
                            "\" >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

[[nodiscard]] uir::FuncId find_function(const uir::Module& m, std::string_view name) {
    for (std::size_t i = 0; i < m.num_functions(); ++i) {
        if (m.function(static_cast<uir::FuncId>(i)).name == name)
            return static_cast<uir::FuncId>(i);
    }
    return uir::INVALID;
}

// Locate the fragment output variable's storage cell by trying the conventional names our
// harness and Shadertoy shaders use. Returns nullptr if none are present.
[[nodiscard]] cpuref::Cell* find_output(cpuref::Interp& interp) {
    static constexpr std::array names{"_O", "outColor", "fragColor", "gl_FragColor", "color", "FragColor"};
    for (std::string_view n : names)
        if (cpuref::Cell* c = interp.global(std::string(n))) return c;
    return nullptr;
}

[[nodiscard]] bool is_finite(const cpuref::Val& v) {
    for (unsigned k = 0; k < v.n; ++k)
        if (!std::isfinite(v.f[k])) return false;
    return true;
}

// Result of running the lifted shader on the CPU reference over a coordinate grid.
struct ExecProbe {
    bool   ran          = false;  // the interpreter executed the function to completion
    bool   all_finite   = true;   // no NaN/Inf in any sampled output
    double output_var   = 0.0;    // luminance variance across the grid (0 == degenerate)
    int    samples      = 0;
};

// Best-effort execution: drive `f` across an NxN grid of fragment coordinates. The v1
// interpreter does not cover every shader (built-ins, unbounded loops); a failure to run is
// reported, never fatal — compile/lift still stand.
[[nodiscard]] ExecProbe probe_execution(const uir::Module& m, uir::FuncId f, int grid, float res) {
    ExecProbe p;
    std::vector<double> lum;
    lum.reserve(static_cast<std::size_t>(grid) * grid);

    for (int gy = 0; gy < grid; ++gy) {
        for (int gx = 0; gx < grid; ++gx) {
            cpuref::Interp interp(m);                 // fresh per-lane state
            // Seed common fragment inputs if the shader exposes them by name.
            const float fx = (static_cast<float>(gx) + 0.5f) / static_cast<float>(grid) * res;
            const float fy = (static_cast<float>(gy) + 0.5f) / static_cast<float>(grid) * res;
            if (cpuref::Cell* fc = interp.global("gl_FragCoord")) fc->val = cpuref::Val::vecf({fx, fy, 0.0f, 1.0f});
            if (cpuref::Cell* fc = interp.global("fragCoord"))    fc->val = cpuref::Val::vecf({fx, fy});

            std::string err;
            if (!interp.run(f, &err)) return p;        // unsupported instruction -> ran stays false
            p.ran = true;

            cpuref::Cell* out = find_output(interp);
            if (out == nullptr || !out->leaf) continue;
            if (!is_finite(out->val)) { p.all_finite = false; }
            const cpuref::Val& c = out->val;
            lum.push_back(0.2126 * c.f[0] + 0.7152 * c.f[1] + 0.0722 * c.f[2]);
            ++p.samples;
        }
    }

    if (lum.size() >= 2) {
        double mean = 0.0;
        for (double x : lum) mean += x;
        mean /= static_cast<double>(lum.size());
        double var = 0.0;
        for (double x : lum) var += (x - mean) * (x - mean);
        p.output_var = var / static_cast<double>(lum.size());
    }
    return p;
}

#if defined(OMNI_HAVE_VULKAN)
// Strip a Shadertoy harness down to the mainImage + helper functions (the compute below
// redeclares the uniforms itself).
[[nodiscard]] std::string mainimage_body(const std::string& src) {
    std::istringstream in(src);
    std::string line, out;
    while (std::getline(in, line)) {
        if (line.find("void main(") != std::string::npos) break;          // stop at harness entry
        if (line.find("#version") != std::string::npos || line.find("layout(") != std::string::npos
            || line.find("push_constant") != std::string::npos || line.find("uniform ") != std::string::npos
            || line.find("#define i") != std::string::npos) continue;     // drop harness decls
        out += line + "\n";
    }
    return out;
}

[[nodiscard]] bool compile_compute(const std::string& glsl, std::vector<uint32_t>& out) {
    std::string comp = "/tmp/omni_rw_XXXXXX.comp";
    std::vector<char> buf(comp.begin(), comp.end()); buf.push_back('\0');
    const int fd = ::mkstemps(buf.data(), 5);
    if (fd == -1) return false;
    ::close(fd);
    comp.assign(buf.data());
    const std::string spv = comp + ".spv";
    { std::ofstream o(comp); o << glsl; }
    const bool ran = std::system(("glslangValidator -V \"" + comp + "\" -o \"" + spv + "\" >/dev/null 2>&1").c_str()) == 0;
    std::remove(comp.c_str());
    std::ifstream in(spv, std::ios::binary);
    if (!ran || !in) { std::remove(spv.c_str()); return false; }
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
    std::remove(spv.c_str());
    out.resize(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
    return !out.empty();
}

// Render mainImage on the REAL GPU and report ran/finite/luminance-variance — the reward term
// a compiler cannot give: a shader can compile and still render NaN or a flat black frame.
struct GpuProbe { bool ran = false; bool finite = true; double variance = 0.0; double chroma = 0.0; int samples = 0; };

[[nodiscard]] GpuProbe gpu_render_probe(const std::string& shader, int W = 96, int H = 96) {
    std::ostringstream src;
    src << "#version 450\nlayout(local_size_x=8, local_size_y=8) in;\n"
        << "layout(std430,set=0,binding=0) buffer Img { vec4 px[]; };\n"
        << "const int WW=" << W << ", HH=" << H << ";\n"
        << "vec3 iResolution=vec3(float(WW),float(HH),1.0);float iTime=1.0;vec4 iMouse=vec4(0.0);"
        << "int iFrame=0;float iTimeDelta=0.016;\n"
        << mainimage_body(shader) << "\n"
        << "void main(){ ivec2 g=ivec2(gl_GlobalInvocationID.xy); if(g.x>=WW||g.y>=HH) return;"
        << " vec2 fragCoord=vec2(g)+0.5; vec4 fc=vec4(0.0,0.0,0.0,1.0); mainImage(fc, fragCoord);"
        << " px[g.y*WW+g.x]=fc; }\n";
    std::vector<std::uint32_t> spirv;
    if (!compile_compute(src.str(), spirv)) return {};
    omni::gpu::VulkanCompute vk;
    std::string err;
    if (!vk.init(&err)) return {};
    const auto raw = vk.run_raw(spirv, static_cast<std::size_t>(W) * H * 16, (W + 7) / 8, (H + 7) / 8, 1);
    if (!raw.ok || raw.bytes.size() < static_cast<std::size_t>(W) * H * 16) return {};
    const float* f = reinterpret_cast<const float*>(raw.bytes.data());
    GpuProbe p; p.ran = true;
    std::vector<double> lum; lum.reserve(static_cast<std::size_t>(W) * H);
    double sat_sum = 0.0;                                  // color richness: mean per-pixel saturation
    for (int i = 0; i < W * H; ++i) {
        const double r = f[i * 4], g = f[i * 4 + 1], b = f[i * 4 + 2];
        for (int c = 0; c < 3; ++c) if (!std::isfinite(f[i * 4 + c])) p.finite = false;
        lum.push_back(0.2126 * r + 0.7152 * g + 0.0722 * b);
        const double s = std::max({r, g, b}) - std::min({r, g, b});   // saturation = chroma span
        sat_sum += std::isfinite(s) ? std::clamp(s, 0.0, 1.0) : 0.0;
    }
    double mean = 0.0; for (double x : lum) mean += x; mean /= static_cast<double>(lum.size());
    double var = 0.0; for (double x : lum) var += (x - mean) * (x - mean);
    p.variance = std::isfinite(var) ? var / static_cast<double>(lum.size()) : 0.0;
    p.chroma = sat_sum / static_cast<double>(lum.size());
    p.samples = static_cast<int>(lum.size());
    return p;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    std::string file;
    int grid = 8;
    float res = 256.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--file" && i + 1 < argc) file = argv[++i];
        else if (a == "--grid" && i + 1 < argc) grid = std::max(1, std::atoi(argv[++i]));
        else if (a == "--res"  && i + 1 < argc) res  = static_cast<float>(std::atof(argv[++i]));
    }

    std::string src;
    if (!file.empty()) {
        std::ifstream is(file, std::ios::binary);
        if (!is) { std::puts(R"({"error":"cannot open --file"})"); return 0; }
        src = read_stream(is);
    } else {
        src = read_stream(std::cin);
    }

    // ---- stage 1: compile ----------------------------------------------------------------
    const synth::GlslValidator validator;
    const synth::CompileResult cr = validator.validate(src, synth::Stage::Fragment);

    reward::Inputs in;
    in.compiled = cr.ok;
    bool lifted = false;
    ExecProbe probe;

    // ---- stages 2-3: lift + execute (only meaningful once it compiles) --------------------
    if (cr.ok && validator.available()) {
        const TempFile frag(".frag");
        const TempFile spv(".spv");
        if (frag.write(src) && emit_spirv(validator.tool_path(), frag.path(), spv.path())) {
            uir::Module m;
            const frontends::SpirvLiftResult lr = frontends::lift_spirv_file(spv.path(), m);
            lifted = lr.ok;
            if (lr.ok) {
                const uir::FuncId f = find_function(m, "main");
                if (f != uir::INVALID) {
                    probe = probe_execution(m, f, grid, res);
                    // Map debugger measurements onto the reward oracle's inputs.
                    in.executed     = probe.ran && probe.all_finite && probe.samples > 0;
                    // A shader that runs to a constant image is "valid but degenerate": give
                    // partial visual credit that saturates with output variance.
                    in.visual_match = in.executed ? (1.0 - std::exp(-probe.output_var * 32.0)) : 0.0;
                }
            }
        }
    }

#if defined(OMNI_HAVE_VULKAN)
    // Prefer the real GPU render for the execute + visual terms — robust on any compiling shader,
    // and the only way to see "compiles but renders NaN / flat black".
    if (cr.ok) {
        const GpuProbe gp = gpu_render_probe(src);
        if (gp.ran) {
            in.executed     = gp.finite;
            // separate, non-saturating structure + color terms so a vivid image beats a grayscale
            // one of the same structure (fights the monochrome failure mode).
            const double structure = 1.0 - std::exp(-gp.variance * 60.0);   // spatial detail
            const double color     = std::min(1.0, gp.chroma * 3.0);        // saturation -> color
            in.visual_match = 0.6 * structure + 0.4 * color;
            probe.ran = true; probe.all_finite = gp.finite;
            probe.output_var = gp.variance; probe.samples = gp.samples;
        }
    }
#endif

    const reward::Breakdown b = reward::score(in);

    // ---- output: one JSON line (sanitize non-finite so it stays valid JSON) ---------------
    const double out_var = std::isfinite(probe.output_var) ? probe.output_var : 0.0;
    std::printf(
        "{\"compiled\":%s,\"lifted\":%s,\"executed\":%s,\"all_finite\":%s,"
        "\"output_variance\":%.6g,\"exec_samples\":%d,\"reward\":%.6f,"
        "\"breakdown\":{\"compile\":%.4f,\"exec\":%.4f,\"visual\":%.4f},"
        "\"compile_log\":\"%s\"}\n",
        in.compiled ? "true" : "false", lifted ? "true" : "false",
        in.executed ? "true" : "false", probe.all_finite ? "true" : "false",
        out_var, probe.samples, b.total,
        b.compile, b.exec, b.visual,
        json_escape(cr.log).c_str());
    return 0;
}
