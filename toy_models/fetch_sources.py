"""Download the pinned public GAMER files used for the source audit.

Network access is required only for this script. Experiments run offline.
"""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

base=Path(__file__).resolve().parent/'sources'
manifest=json.loads((base/'gamer_provenance.json').read_text())
sha=manifest['commit']

def fetch(path):
    target=base/'gamer'/path
    target.parent.mkdir(parents=True,exist_ok=True)
    request=urllib.request.Request(f'https://raw.githubusercontent.com/gamer-project/gamer/{sha}/{path}',
                                  headers={'User-Agent':'pinn-tensorflow-toy-research'})
    target.write_bytes(urllib.request.urlopen(request,timeout=40).read())
    return path

if __name__=='__main__':
    paths=list(dict.fromkeys(manifest['paths']+['LICENSE']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        downloaded=list(pool.map(fetch,paths))
    print(f'Downloaded {len(downloaded)} files at GAMER {sha}')
