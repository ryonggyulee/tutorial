"""Run the tutorial SIESTA input and check completion."""
import argparse
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--siesta', type=Path, required=True)
p.add_argument('--non-scf', action='store_true')
a = p.parse_args()
d = a.directory.resolve()
if (d / 'stdout.txt').exists():
    raise FileExistsError('Use a fresh run directory; stdout.txt already exists.')
started = time.monotonic()
with (d / 'RUN.fdf').open() as inp, (d / 'stdout.txt').open('w') as out, (d / 'stderr.txt').open('w') as err:
    result = subprocess.run([str(a.siesta.resolve())], stdin=inp, stdout=out, stderr=err, cwd=d)
log = (d / 'stdout.txt').read_text(errors='replace')
if result.returncode or 'Job completed' not in log:
    print(log[-5000:])
    print((d / 'stderr.txt').read_text(errors='replace')[-3000:])
    raise RuntimeError('SIESTA did not complete. Check stdout.txt and stderr.txt.')
iterations = len(re.findall(r'^\s*scf:', log, flags=re.M))
if a.non_scf:
    assert iterations == 1, 'Expected exactly one SCF iteration.'
else:
    assert re.search(r'SCF cycle converged', log, re.I), 'Reference SCF did not converge.'
    assert 'SCF_NOT_CONV' not in log, 'Reference SCF failed to converge.'
for name in ('Molecular.RHO', 'Molecular.bands', 'Molecular.EIG', 'Molecular.FA'):
    assert (d / name).is_file() and (d / name).stat().st_size > 0, name
if not a.non_scf:
    for name in ('Molecular.IOCH', 'rhoatom', 'rhoover'):
        assert (d / name).is_file() and (d / name).stat().st_size > 0, name
print(f'Completed: {iterations} SCF iteration(s), {time.monotonic()-started:.1f} s')
print('Log:', d / 'stdout.txt')
