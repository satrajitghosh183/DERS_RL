#!/usr/bin/env python3
"""Generate a debugger-verified shader DATASET + gallery with the best model + best-of-N. For each
prompt: sample K candidates, score each with the OmniTrace oracle, keep the best (highest reward),
render it to a PNG. Output: dataset.jsonl (prompt, glsl, reward, compiled, run, png) + renders/*.png.
A reusable release artifact that can't be regenerated without the GPU.

  python3 gen_dataset.py --base <hf> --adapter <dir> --prompts <jsonl> --reward-bin <b> --render-bin <r> \
      --k 6 --out <dir>
"""
import argparse, json, os, subprocess, tempfile

HARNESS=("#version 450\nlayout(location=0) out vec4 _O;\n"
 "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
 "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n#define iFrame u.iFrame\n")
def wrap(c): return HARNESS+c+"\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
def extract(t):
    if "```" in t:
        seg=t.split("```",2); b=seg[1] if len(seg)>1 else t
        for lg in ("glsl","c","cpp","c++"):
            if b.startswith(lg): b=b[len(lg):]; break
        return b.strip()
    return t.strip()
def omni(rb,glsl):
    try:
        p=subprocess.run([rb],input=glsl,capture_output=True,text=True,errors="replace",timeout=40)
        j=json.loads(p.stdout.strip().splitlines()[-1])
        return bool(j.get("compiled")),bool(j.get("executed")),float(j.get("reward",0))
    except Exception: return False,False,0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--adapter",default=None)
    ap.add_argument("--prompts",required=True); ap.add_argument("--reward-bin",required=True)
    ap.add_argument("--render-bin",required=True); ap.add_argument("--k",type=int,default=6)
    ap.add_argument("--max-new",type=int,default=384); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    os.makedirs(a.out+"/renders",exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,device_map="auto")
    if a.adapter:
        from peft import PeftModel; m=PeftModel.from_pretrained(m,a.adapter)
    m.eval()
    prompts=[json.loads(l)["prompt"] for l in open(a.prompts) if l.strip()]
    print(f"== dataset: {len(prompts)} prompts, best-of-{a.k} ==",flush=True)
    ds=open(a.out+"/dataset.jsonl","w"); kept=0; rendered=0
    for i,pr in enumerate(prompts):
        ids=tok(f"// Shader: {pr}\n",return_tensors="pt").to(m.device)
        with torch.no_grad():
            g=m.generate(**ids,max_new_tokens=a.max_new,do_sample=True,temperature=0.7,top_p=0.9,
                         repetition_penalty=1.1,num_return_sequences=a.k,pad_token_id=tok.eos_token_id)
        cands=[extract(tok.decode(s[ids.input_ids.shape[1]:],skip_special_tokens=True)) for s in g]
        scored=[(c,)+omni(a.reward_bin,wrap(c)) for c in cands]
        best=max(scored,key=lambda x:x[3])          # (code, compiled, run, reward)
        code,comp,run,rew=best
        png=""
        if comp:
            slug="".join(ch if ch.isalnum() else "_" for ch in pr)[:32]
            raw=f"{a.out}/renders/{i:04d}_{slug}.glsl"; open(raw,"w").write(code)
            ppm=raw[:-5]+".ppm"
            subprocess.run([a.render_bin,raw,ppm,"256","256","1.0"],capture_output=True,timeout=30)
            if os.path.exists(ppm): png=os.path.basename(ppm); rendered+=1
            else: os.remove(raw) if os.path.exists(raw) else None
        ds.write(json.dumps({"prompt":pr,"glsl":code,"compiled":comp,"runs":run,"reward":round(rew,2),"render":png})+"\n")
        kept+=int(comp)
        if (i+1)%25==0: print(f"  {i+1}/{len(prompts)}  compiled={kept} rendered={rendered}",flush=True)
    ds.close()
    print(f"== DONE: {len(prompts)} prompts, {kept} compiled, {rendered} rendered -> {a.out} ==",flush=True)
    json.dump({"n":len(prompts),"compiled":kept,"rendered":rendered},open(a.out+"/stats.json","w"))

if __name__=="__main__": main()
