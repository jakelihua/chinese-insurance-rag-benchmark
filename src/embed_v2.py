"""Pinned BGE CLS embeddings with explicit truncation counts and cache hashes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from experiment import write_json

INSTRUCTION = '为这个句子生成表示以用于检索相关文章：'


class Encoder:
    def __init__(self, path, batch_size=32):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True).to(self.device).eval()

    def encode(self, texts, query=False):
        if not texts:
            raise ValueError('Cannot encode an empty query collection')
        vectors, truncated = [], 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if query:
                batch = [INSTRUCTION + text for text in batch]
            lengths = self.tokenizer(batch, truncation=False, return_length=True)['length']
            truncated += sum(n > 512 for n in lengths)
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512,
                                     return_tensors='pt').to(self.device)
            with torch.inference_mode():
                cls = self.model(**encoded).last_hidden_state[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            array = cls.cpu().numpy()
            if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
                raise ValueError('Invalid embedding values or normalization')
            vectors.append(array)
            if i % 512 == 0:
                print('Embedded', i, '/', len(texts), 'query', query, flush=True)
        return np.concatenate(vectors), truncated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(args.inputs.read_text())
    started = time.time()
    encoder = Encoder(args.model)
    results, truncation = {}, {}
    for name, texts, query in [
        ('documents', inputs['documents'], False),
        ('real_calibration', [x['question'] for x in inputs['calibration']], True),
        ('real_test', [x['question'] for x in inputs['test']], True),
    ]:
        results[name], truncation[name] = encoder.encode(texts, query)
    np.savez_compressed(args.out / 'embeddings.npz', **results)
    write_json(args.out / 'manifest.json', {
        'evaluation_inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'model_artifacts': json.loads((args.model / 'artifact_manifest.json').read_text()),
        'device': encoder.device, 'dtype': 'float32', 'max_length': 512, 'pooling': 'CLS, L2-normalized',
        'query_instruction': INSTRUCTION, 'truncated': truncation,
        'shapes': {k: list(v.shape) for k, v in results.items()},
        'elapsed_seconds': time.time() - started,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })


if __name__ == '__main__':
    main()
