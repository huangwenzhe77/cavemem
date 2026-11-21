import json
import os
import sys
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

weights = os.environ["CAVE_QWEN_WEIGHTS"]
tok=AutoTokenizer.from_pretrained(weights,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(weights,torch_dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
for line in sys.stdin:
    req=json.loads(line)
    ids=tok(req['prompt'],return_tensors='pt').input_ids.to('cuda')
    kwargs={'do_sample':req['seed'] is not None}
    if req['seed'] is not None:
        torch.manual_seed(req['seed']);kwargs.update(temperature=.7,top_p=.95)
    start=perf_counter()
    with torch.inference_mode():
        token=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=1152,
             pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,stop_strings=['\n```'],tokenizer=tok,**kwargs)[0,ids.shape[1]:]
    print(json.dumps({'text':tok.decode(token,skip_special_tokens=True),'seconds':perf_counter()-start,
                      'input_tokens':int(ids.shape[1]),'output_tokens':int(token.numel())}),flush=True)
