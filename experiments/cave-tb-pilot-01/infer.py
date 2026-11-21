"""Private model worker. Stdin and stdout carry public prompts and metered completions."""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

weights = os.environ["CAVE_QWEN_WEIGHTS"]
tok = AutoTokenizer.from_pretrained(weights, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=torch.bfloat16,
                                            local_files_only=True).to("cuda").eval()
for line in sys.stdin:
    req = json.loads(line)
    ids = tok.apply_chat_template(req["messages"], tokenize=True,
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    opts = {}
    if req.get("seed") is not None:
        torch.manual_seed(req["seed"])
        opts.update(temperature=.7, top_p=.95)
    t = time.monotonic()
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=bool(opts),
                             max_new_tokens=1024, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
                             stop_strings=["</action>"], tokenizer=tok, **opts)
    tail = out[0,ids.shape[1]:]
    print(json.dumps({"text":tok.decode(tail, skip_special_tokens=True),
                      "input_tokens":ids.shape[1],"output_tokens":tail.numel(),
                      "seconds":time.monotonic()-t}),flush=True)
