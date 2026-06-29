import sys, json, subprocess, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, "/home/exouser/shader_pipeline")
from eval_batched import extract_code, wrap
base, adapter, name = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [json.loads(l)["prompt"] for l in open(sys.argv[4] if len(sys.argv)>4 else "/home/exouser/shader_data/rl_prompts.jsonl") if l.strip()][:20]
tok = AutoTokenizer.from_pretrained(base)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
m = PeftModel.from_pretrained(m, adapter); m.eval()
OMNI="/home/exouser/NERC/build/omni_reward"
ok=0
for p in prompts:
    ids=tok(p,return_tensors="pt").to(m.device)
    g=m.generate(**ids,max_new_tokens=256,do_sample=True,temperature=0.7,top_p=0.95,repetition_penalty=1.1,pad_token_id=tok.eos_token_id)
    code=extract_code(tok.decode(g[0][ids.input_ids.shape[1]:],skip_special_tokens=True))
    try:
        j=json.loads(subprocess.run([OMNI],input=wrap(code),capture_output=True,text=True,timeout=30).stdout.strip().splitlines()[-1])
        ok+=int(j.get("compiled",False))
    except: pass
print(f"{name} compile@1 = {ok}/{len(prompts)} = {ok/len(prompts):.3f}")
open(f"/home/exouser/pipeline/{name}_ba.json","w").write(json.dumps({"name":name,"compiles":ok,"n":len(prompts),"compile_at_1":ok/len(prompts)}))
