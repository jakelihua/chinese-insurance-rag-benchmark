"""Download pinned public model artifacts; no remote Python code or credentials."""
import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

MODELS = {
    'qwen': ('Qwen/Qwen2.5-1.5B-Instruct', '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
             ['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
              'merges.txt', 'vocab.json', 'model.safetensors', 'LICENSE']),
    'bge': ('BAAI/bge-small-zh-v1.5', '7999e1d3359715c523056ef9478215996d62a620',
            ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.txt', 'model.safetensors']),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, destination):
    part = destination.with_name(destination.name + '.part')
    subprocess.run(['curl', '-sS', '-L', '--fail', '--retry', '3', '--retry-all-errors',
                    '--max-time', '600', url, '-o', str(part)], check=True)
    part.replace(destination)


def download(tag, root):
    repo, revision, files = MODELS[tag]
    destination = root / tag
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / 'upstream_metadata.json'
    fetch(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', metadata_path)
    metadata = json.loads(metadata_path.read_text())
    upstream = {a['rfilename']: a for a in metadata['siblings']}
    if metadata['sha'] != revision:
        raise ValueError('Model revision mismatch')

    def one(filename):
        path = destination / filename
        info = upstream[filename]
        expected_sha = info.get('lfs', {}).get('sha256')
        expected_size = info.get('size')
        valid = path.exists() and (expected_size is None or path.stat().st_size == expected_size)
        if valid and expected_sha:
            valid = sha256(path) == expected_sha
        if not valid:
            print('Downloading', tag, filename, flush=True)
            fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true', path)
        digest = sha256(path)
        if expected_sha and digest != expected_sha:
            raise ValueError(f'Checksum mismatch: {tag}/{filename}')
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f'Size mismatch: {tag}/{filename}')
        print('Verified', tag, filename, path.stat().st_size, flush=True)
        return {'file': filename, 'bytes': path.stat().st_size, 'sha256': digest,
                'upstream_sha256_verified': expected_sha is not None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, files))
    (destination / 'artifact_manifest.json').write_text(json.dumps(
        {'repo': repo, 'revision': revision, 'files': records}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda tag: download(tag, args.root), MODELS))
