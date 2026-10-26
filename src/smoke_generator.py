"""Numerical smoke checks before an expensive generation run."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_v2 import SYSTEM_PROMPT, USER_PROMPT, parse_question


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.inputs.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side='left')
    texts = []
    for row in data['sources'][:4]:
        short = tokenizer.decode(tokenizer.encode(row['answer'], add_special_tokens=False)[:512])
        texts.append(tokenizer.apply_chat_template(
            [{'role':'system','content':SYSTEM_PROMPT},
             {'role':'user','content':USER_PROMPT.format(source=short)}],
            tokenize=False, add_generation_prompt=True))
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('mps').eval()
    encoded = tokenizer(texts, padding=True, return_tensors='pt').to('mps')
    with torch.inference_mode():
        logits = model(**encoded).logits[:, -1].float()
        finite = torch.isfinite(logits).all().item()
        print('Final-position logits finite:', finite, flush=True)
        if not finite:
            raise ValueError('Nonfinite logits in float32 eager smoke')
        generated = model.generate(**encoded, max_new_tokens=64, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tokenizer.pad_token_id)
    outputs = tokenizer.batch_decode(generated[:, encoded['input_ids'].shape[1]:], skip_special_tokens=True)
    result = {'dtype':'float32','attention':'eager','device':'mps','logits_finite':finite,
              'elapsed_seconds':time.time()-started,
              'outputs':[{'raw':s,'format_errors':parse_question(s)[1]} for s in outputs]}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
