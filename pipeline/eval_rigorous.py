#!/usr/bin/env python3
"""Rigorous, publication-grade evaluation. For each model checkpoint, sample K completions per
held-out prompt and score every one with the OmniTrace oracle (compile + run + render). Reports:
  compile@k for k=1..K  (any-of-k builds)         -> the curve reviewers want
  run@k, render@k                                  -> execution-grounded quality
  best-of-N (debugger-picked) usable@1             -> the inference-time lift
  mean best-of-K reward                            -> overall quality
Writes a JSON per model + a combined table; plots are drawn by plot_eval.py.

  python3 eval_rigorous.py --base <hf> [--adapter <dir>] --prompts <jsonl> --reward-bin <bin> \
      --k 8 --label <name> --out <json>
"""
import argparse, json, os, subprocess, sys, tempfile

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n#define iFrame u.iFrame\n")
def wrap(c):
    if "mainImage" in c:
        return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    return HARNESS + "void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy; vec3 col=vec3(0.);\n"+c+"\n_O=vec4(col,1.); }\n"
def extract(t):
    if "```" in t:
        seg=t.split("```",2); b=seg[1] if len(seg)>1 else t
        for lg in ("glsl","c","cpp","c++"):
            if b.startswith(lg): b=b[len(lg):]; break
        return b.strip()
    return t.strip()
def score(rb, glsl, thr=0.001):
    try:
        p=subprocess.run([rb],input=glsl,capture_output=True,text=True,errors="replace",timeout=40)
        j=json.loads(p.stdout.strip().splitlines()[-1])
        return bool(j.get("compiled")), bool(j.get("executed")), float(j.get("output_variance",0))>thr, float(j.get("reward",0))
    except Exception:
        return False,False,False,0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",required=True); ap.add_argument("--adapter",default=None)
    ap.add_argument("--prompts",required=True); ap.add_argument("--reward-bin",required=True)
    ap.add_argument("--k",type=int,default=8); ap.add_argument("--max-new",type=int,default=384)
    ap.add_argument("--temp",type=float,default=0.7); ap.add_argument("--label",default="model")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.base,dtype=torch.bfloat16,device_map="auto")
    if a.adapter:
        from peft import PeftModel; m=PeftModel.from_pretrained(m,a.adapter)
    m.eval()
    prompts=[json.loads(l) for l in open(a.prompts) if l.strip()]
    K=a.k
    print(f"== {a.label}: {len(prompts)} prompts x k={K} ==",flush=True)
    comp_at=[0]*(K+1); run1=rend1=0; bestN_usable=0; sumbest=0.0
    rows=[]
    for i,p in enumerate(prompts):
        tp=p["prompt"] if p["prompt"].startswith("//") else f"// Shader: {p['prompt']}"
        ids=tok(tp+"\n",return_tensors="pt").to(m.device)
        with torch.no_grad():
            g=m.generate(**ids,max_new_tokens=a.max_new,do_sample=True,temperature=a.temp,top_p=0.9,
                         repetition_penalty=1.1,num_return_sequences=K,pad_token_id=tok.eos_token_id)
        scored=[score(a.reward_bin, wrap(extract(tok.decode(s[ids.input_ids.shape[1]:],skip_special_tokens=True)))) for s in g]
        comp=[s[0] for s in scored]; ex=[s[1] for s in scored]; rnd=[s[2] for s in scored]; rew=[s[3] for s in scored]
        for k in range(1,K+1): comp_at[k]+=int(any(comp[:k]))    # compile@k
        run1+=int(ex[0]); rend1+=int(rnd[0])
        # best-of-N: pick the candidate with the HIGHEST debugger reward; "usable" if it runs clean
        bi=max(range(K),key=lambda j:rew[j]); bestN_usable+=int(ex[bi]); sumbest+=rew[bi]
        rows.append({"prompt":tp,"compile":comp,"run":ex,"render":rnd,"reward":[round(x,2) for x in rew]})
        print(f"  [{i+1}/{len(prompts)}] c@1={int(comp[0])} c@k={int(any(comp))} bestN_run={int(ex[bi])} maxR={max(rew):.2f}",flush=True)
    N=len(prompts)
    res={"label":a.label,"n":N,"k":K,
         "compile@k":{str(k):round(comp_at[k]/N,3) for k in range(1,K+1)},
         "run@1":round(run1/N,3),"render@1":round(rend1/N,3),
         "bestN_usable@1":round(bestN_usable/N,3),"mean_bestK_reward":round(sumbest/N,3)}
    print(f"\n== {a.label} ==\n  compile@1={res['compile@k']['1']} compile@{K}={res['compile@k'][str(K)]}")
    print(f"  bestN_usable@1={res['bestN_usable@1']}  mean_bestK_reward={res['mean_bestK_reward']}")
    json.dump({"summary":res,"rows":rows},open(a.out,"w"),indent=2)
    print(f"-> {a.out}",flush=True)

if __name__=="__main__": main()
