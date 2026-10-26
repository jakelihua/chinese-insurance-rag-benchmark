"""Pinned 4-bit derivative for the memory-limited third-round experiment."""
from pathlib import Path
import argparse

import download_models


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    download_models.MODELS['qwen_mlx']=(
        'mlx-community/Qwen2.5-1.5B-Instruct-4bit','8b403126fc14f14cfc99bb4cfa72ecbc129ea677',
        ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json',
         'added_tokens.json','merges.txt','vocab.json','model.safetensors','model.safetensors.index.json','README.md'])
    download_models.download('qwen_mlx',args.root)
