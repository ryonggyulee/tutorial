"""Predict with the repository model and preserve the reference RHO header."""
import argparse
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('project', type=Path)
p.add_argument('work', type=Path)
p.add_argument('--case', default='polymer')
p.add_argument('--label', default='Molecular')
a = p.parse_args()
project, work = a.project.resolve(), a.work.resolve()
sys.path.insert(0, str(project))
from model import ResUNet
from utils.siestaio import readGrid, writeGrid

config = OmegaConf.load(str(project / 'configs/predict/examples/r_plus_s_x3.yaml'))
scale = {key.strip(): float(value) for key, value in
         (line.split('=') for line in (project / 'datasets/scale/scale_r_plus_s_x3.txt').read_text().splitlines() if line.strip())}
assert float(scale['mean3']) == config.model.mean
assert float(scale['std3']) == config.model.std
# Both examples use CPU inference and the same pretrained normalization.
torch.set_num_threads(2)
scf = work / 'scf' / a.case
dataset = work / 'dataset' / (a.case + '.h5')
cell, mesh, reference = readGrid(str(scf / (a.label + '.RHO')))
assert reference.shape == (1, *mesh)
assert np.all(mesh % 8 == 0)
with h5py.File(str(dataset), 'r') as f:
    feature, target = f['feature'][:], f['target'][:]
assert feature.shape == (3, *mesh) and target.shape == tuple(mesh)
assert np.isfinite(feature).all() and np.isfinite(target).all()
assert np.allclose(target, reference[0])
print('feature:', feature.shape, '| target:', target.shape, flush=True)
model = ResUNet(config.model).cpu()
checkpoint = torch.load(str(project / config.load.model), map_location='cpu')
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
model.eval()
start = time.monotonic()
with torch.no_grad():
    rho = model(torch.from_numpy(feature).float().unsqueeze(0)).numpy()
assert np.isfinite(rho).all()
output = work / 'predict'
output.mkdir(exist_ok=True)
prediction = output / (a.case + '.RHO')
writeGrid(str(prediction), cell, mesh, rho)
# Check the saved density, not a second inference.
c2, m2, r2 = readGrid(str(prediction))
assert np.allclose(c2, cell) and np.array_equal(m2, mesh)
assert np.allclose(r2, rho)
initial = feature[1] * config.model.std + config.model.mean
cases = [('ML', r2[0])] if a.case != 'polymer' else [('Atomic', initial), ('ML', r2[0])]
for label, values in cases:
    error = 100 * np.abs(values - target).sum() / target.sum()
    print('%s density error: %.6f %%' % (label, error))
print('Prediction time: %.1f s' % (time.monotonic()-start))
print('Saved:', prediction)
