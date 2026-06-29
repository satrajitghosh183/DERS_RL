#!/usr/bin/env python3
"""Parametrized DoRA fine-tune (PEFT use_dora=True), BF16 across available GPUs,
then generate + compile@k. Works for any HF causal-LM base (Qwen-Coder-32B or -3B).
Usage: finetune_dora.py <model_dir> <steps> <out_name>"""
import torch, os, glob, subprocess, tempfile, sys, json
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

MODEL = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
NAME  = sys.argv[3] if len(sys.argv) > 3 else "dora"
import sys as _s
TEXTS = _s.argv[4] if len(_s.argv)>4 else os.path.expanduser("~/shader_data/texts")
OUT   = os.path.expanduser(f"~/pipeline/{NAME}_out"); os.makedirs(OUT, exist_ok=True)
GLSLANG = "/usr/bin/glslangValidator"
def log(*a): print(*a, flush=True)
log(f"== DoRA fine-tune: {MODEL} | {STEPS} steps | out={NAME} ==")

docs = []
for f in glob.glob(TEXTS + "/*.txt"):
    for d in open(f).read().split("<|endoftext|>"):
        d = d.strip()
        if len(d) > 50: docs.append(d)
log(f"shader docs: {len(docs)}")

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = Dataset.from_dict({"text": docs}).map(
    lambda e: tok(e["text"], truncation=True, max_length=1024),
    batched=True, remove_columns=["text"])

log("loading base in BF16 across GPUs (SDPA)...")
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
        device_map="auto", max_memory={0: "74GiB", 1: "74GiB"}, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", use_dora=True))
model.print_trainable_parameters()

Trainer(model=model,
        args=TrainingArguments(output_dir=OUT, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, max_steps=STEPS, learning_rate=1e-4, bf16=True,
            logging_steps=10, save_steps=100000, report_to=[], dataloader_num_workers=4,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
model.save_pretrained(OUT + "/adapter")
log("adapter saved")

HARNESS = ("#version 450\nlayout(location=0) out vec4 _O;\n"
  "layout(push_constant) uniform U { vec3 iResolution; float iTime; vec4 iMouse; int iFrame; } u;\n"
  "#define iResolution u.iResolution\n#define iTime u.iTime\n#define iMouse u.iMouse\n")
def wrap(c):
    if "mainImage" in c:
        return HARNESS + c + "\nvoid main(){ vec4 c=vec4(0.); mainImage(c, gl_FragCoord.xy); _O=c; }\n"
    return HARNESS + ("void main(){ vec2 fragCoord=gl_FragCoord.xy; vec2 uv=fragCoord/iResolution.xy;"
                      " vec3 col=vec3(0.);\n" + c + "\n_O=vec4(col,1.); }\n")
def valid(g):
    f=tempfile.NamedTemporaryFile(suffix=".frag",delete=False,mode="w"); f.write(g); f.close()
    try: return subprocess.run([GLSLANG,"-V",f.name,"-o","/dev/null"],capture_output=True,timeout=20).returncode==0
    finally: os.unlink(f.name)

model.eval()
prompts=["// Shader: blue fire","// Shader: ocean waves","// Shader: red plasma","// Shader: starfield sky",
         "// Shader: rainbow gradient","// Shader: noise clouds","// Shader: rotating tunnel","// Shader: water ripples"]
log(f"\n== {NAME} generation + compile@k ==")
comp=0; samples=[]
for p in prompts:
    ids = tok(p+"\n", return_tensors="pt").to(model.device)
    g = model.generate(**ids, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9,
                       repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    text = tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    ok = len(text.strip())>30 and valid(wrap(text))
    comp += int(ok); samples.append((p, ok, text))
    log(f"  {p:28s} compiles={int(ok)}  gen[:60]={text.strip()[:60]!r}")
res = comp/len(prompts)
log(f"\n{NAME} compile@1 = {comp}/{len(prompts)} = {res:.2f}")
json.dump({"name": NAME, "model": MODEL, "compile_at_1": res, "compiled": comp, "n": len(prompts)},
          open(OUT+"/metrics.json","w"))
with open(OUT+"/samples.txt","w") as f:
    for p,ok,t in samples: f.write(f"### {p}  (compiles={ok})\n{t}\n\n")
log(f"metrics -> {OUT}/metrics.json ; samples -> {OUT}/samples.txt\nDONE")
