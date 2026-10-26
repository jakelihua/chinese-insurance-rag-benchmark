"""Cloud-API Round 2 query generation (zero_shot & few_shot_3) for multi-scale retriever selection validation."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiment import write_json
from generate_v2 import SYSTEM_PROMPT, USER_PROMPT, parse_question


def call_chat_api(api_url, api_key, model_name, system_msg, user_msg, max_retries=5):
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ],
        'temperature': 0.0,
        'max_tokens': 64,
    }
    body = json.dumps(payload).encode('utf-8')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    for attempt in range(max_retries):
        tick = time.time()
        req = urllib.request.Request(api_url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                elapsed = time.time() - tick
                choice = data['choices'][0]
                raw = choice['message']['content']
                finish_reason = choice.get('finish_reason', 'stop')
                usage = data.get('usage', {})
                return raw, finish_reason, usage, elapsed
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--model-name', type=str, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--api-url', type=str, default='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()

    api_key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('DASHSCOPE_API_KEY environment variable is required')

    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    if set(inputs) != {'seed', 'demonstration_questions', 'sources'}:
        raise ValueError('Generation accepts only the frozen generation input schema')
    if len(inputs['demonstration_questions']) != 3:
        raise ValueError('Exactly 3 calibration demonstrations required')

    args.out.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    started = time.time()

    modes = ['zero_shot', 'few_shot_3']
    jobs = []
    for source in inputs['sources']:
        shortened = source['answer'][:800]
        for mode in modes:
            style = ''
            if mode == 'few_shot_3':
                style = (
                    '以下是真实用户问法示例，仅参考表达风格，不要复制其内容：\n' +
                    '\n'.join(inputs['demonstration_questions']) + '\n\n'
                )
            prompt = style + USER_PROMPT.format(source=shortened)
            jobs.append({
                'order': len(jobs),
                'doc_id': source['doc_id'],
                'mode': mode,
                'prompt': prompt,
                'source_tokens_original': len(source['answer']),
                'source_truncated': len(source['answer']) > 800,
            })

    results = [None] * len(jobs)

    def run_one(job):
        raw, finish_reason, usage, elapsed = call_chat_api(
            args.api_url, api_key, args.model_name, SYSTEM_PROMPT, job['prompt']
        )
        capped = (finish_reason == 'length')
        question, errors = parse_question(raw, capped=capped)
        return {
            'doc_id': job['doc_id'],
            'mode': job['mode'],
            'source_tokens_original': job['source_tokens_original'],
            'source_truncated': job['source_truncated'],
            'raw_output': raw,
            'question': question,
            'format_valid': not errors,
            'errors': errors,
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'generated_tokens': usage.get('completion_tokens', 0),
            'batch_seconds': elapsed,
            'prompt_sha256': hashlib.sha256(job['prompt'].encode('utf-8')).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(run_one, j): j['order'] for j in jobs}
        done_count = 0
        for fut in concurrent.futures.as_completed(future_map):
            order = future_map[fut]
            results[order] = fut.result()
            done_count += 1
            if done_count % 50 == 0 or done_count == len(jobs):
                print(f"[{args.model_name}] Round2 Generated {done_count}/{len(jobs)} elapsed {round(time.time()-started, 1)}s", flush=True)

    raw_path = args.out / 'generations.jsonl'
    with raw_path.open('w', encoding='utf-8') as log:
        for r in results:
            log.write(json.dumps(r, ensure_ascii=False) + '\n')

    write_json(args.out / 'manifest.json', {
        'input_sha256': input_hash,
        'model': args.model_name,
        'revision': 'dashscope-api',
        'seed': inputs['seed'],
        'device': 'cloud-api',
        'dtype': 'api-default',
        'generation_count': len(jobs),
        'elapsed_seconds': time.time() - started,
        'system_prompt': SYSTEM_PROMPT,
        'user_prompt_template': USER_PROMPT,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


if __name__ == '__main__':
    main()
