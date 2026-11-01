"""Fetch/cache the pinned public sources required by the Gram-FE audit."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
SHA=json.loads((ROOT/'sources/gamer_provenance.json').read_text())['commit']
PATHS=[
    'LICENSE','include/Macro.h','include/GramFE_ExtensionTables.h',
    'src/Model_ELBDM/ELBDM_GramFE_EvolutionMatrix.cpp',
    'src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp',
    'src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_MATMUL.cpp',
    'src/Model_ELBDM/ELBDM_GetTimeStep_Fluid.cpp',
    'src/Model_ELBDM/ELBDM_GetTimeStep_Gravity.cpp',
    'src/Main/InterpolateGhostZone.cpp','src/Init/Init_ResetParameter.cpp',
    'src/configure.py','example/test_problem/ELBDM/LSS_Hybrid_Zoomin/Input__Parameter',
    'doc/wiki/Runtime-Parameters-related/[Runtime-Parameters]-ELBDM.md',
]


if __name__=='__main__':
    records=[]
    for path in PATHS:
        dest=ROOT/'sources/gamer'/path
        url=f'https://raw.githubusercontent.com/gamer-project/gamer/{SHA}/{path}'
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        records.append(dict(path=path,url=url,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    out=ROOT/'gramfe_results';out.mkdir(exist_ok=True)
    (out/'source_manifest.json').write_text(json.dumps(dict(commit=SHA,files=records),indent=2)+'\n')
    print('Recorded hashes for cached/fetched source files:',len(records))
