import json, os, glob, math
SRC = os.path.expanduser("~/shader_data/codes/shader_codes")
OUT = os.path.expanduser("~/shader_data/texts"); os.makedirs(OUT, exist_ok=True)
docs = []
for f in glob.glob(SRC + "/shadertoy/**/*.fragment", recursive=True):
    try:
        j = json.load(open(f))
        info = j.get("info", {}) or {}
        name = (info.get("name") or "").strip()
        desc = (info.get("description") or "").strip().replace("\n", " ")
        tags = info.get("tags") or []
        code = "\n\n".join(p.get("code","") for p in (j.get("renderpass") or []) if p.get("code"))
        if not code.strip(): continue
        hdr = "// Shader: " + name + "\n"
        if desc: hdr += "// " + desc + "\n"
        if tags: hdr += "// tags: " + ", ".join(str(t) for t in tags) + "\n"
        docs.append(hdr + code.strip() + "\n")
    except Exception:
        continue
SEP = "\n<|endoftext|>\n"
N = 40; per = max(1, math.ceil(len(docs)/N))
for i in range(N):
    chunk = docs[i*per:(i+1)*per]
    if not chunk: break
    open(OUT + "/shard_%03d.txt" % i, "w").write(SEP.join(chunk))
total_chars = sum(len(d) for d in docs)
print("shaders:", len(docs), "| chars:", total_chars, "| ~est tokens:", total_chars//4, "| ->", OUT)
