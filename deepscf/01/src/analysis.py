"""Plotting helpers for the polymer and DNA examples."""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import sisl


def density(path):
    # rhoatom and *.RHO share SIESTA's binary grid format.
    reader = type(sisl.get_sile('dummy.RHO'))
    grid = reader(str(path)).read_grid()
    assert np.isfinite(grid.grid).all()
    return grid  # sisl returns electron density in e / Ang^3.


def plot_density(initial=None, ml=None, scf=None):
    """Plot density files in initial, ML, SCF order; scf is required."""
    ref = density(scf)
    grids = [('SCF', ref)]
    if initial is not None:
        grids.insert(0, ('Atomic', density(initial)))
    if ml is not None:
        grids.insert(-1, ('DeepSCF', density(ml)))
    for _, grid in grids:
        assert grid.shape == ref.shape
        assert np.allclose(grid.cell, ref.cell)
    cell = np.diag(ref.cell)
    assert np.allclose(ref.cell, np.diag(cell))
    dy = cell[1] / ref.shape[1]
    projected = [(label, g.grid.sum(axis=1) * dy) for label, g in grids]
    if ml is not None:
        diff = density(ml).grid - ref.grid
        projected.append(('DeepSCF - SCF', diff.sum(axis=1) * dy))
    fig, axes = plt.subplots(1, len(projected), figsize=(4*len(projected), 4), constrained_layout=True, squeeze=False)
    vmax = max(values.max() for label, values in projected if label != 'DeepSCF - SCF')
    for ax, (label, values) in zip(axes[0], projected):
        delta = label == 'DeepSCF - SCF'
        limit = max(abs(values.min()), abs(values.max()), 1e-12) if delta else vmax
        im = ax.imshow(values.T, origin='lower', extent=(0, cell[0], 0, cell[2]),
                       cmap='RdBu_r' if delta else 'viridis', vmin=-limit if delta else 0, vmax=limit,
                       aspect='equal')
        ax.set(xlabel='x (Angstrom)', ylabel='z (Angstrom)', title=label, xlim=(7,13))
        fig.colorbar(im, ax=ax, label=r'$\int\rho\,dy$ (e / $\mathrm{\AA}^2$)', shrink=0.8)
    electrons = ref.grid.sum() * ref.dvolume
    print('Integrated SCF charge: %.6f electrons' % electrons)
    for label, grid in grids:
        if label != 'SCF':
            error = 100 * np.abs(grid.grid-ref.grid).sum() / ref.grid.sum()
            print('%s density error (full 3D grid): %.6f %%' % (label,error))
    plt.show()
    return fig


def read_bands(directory):
    reader = sisl.get_sile(str(Path(directory) / 'Molecular.bands'))
    ticks, k, values = reader.read_data()
    assert values.shape[1] == 1, 'This example is non-spin-polarized.'
    return np.asarray(k), values[:,0,:] + reader.read_fermi_level()


def plot_bands(initial=None, ml=None, scf=None):
    """Plot run directories in initial, ML, SCF order; scf is required."""
    k, reference = read_bands(scf)
    # C8H16: 48 valence electrons, 24 doubly occupied bands.
    nocc = 24
    zero = reference[:, nocc-1].max()
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    cases = [('ρ_SCF', k, reference, 'black', '-')]
    if ml is not None:
        km, ml_bands = read_bands(ml)
        assert ml_bands.shape == reference.shape and np.allclose(km,k)
        cases.insert(0, ('ρ_ML', km, ml_bands, 'tab:red', '--'))
    if initial is not None:
        ki, ei = read_bands(initial)
        assert ei.shape == reference.shape and np.allclose(ki, k)
        cases.insert(0, ('ρ_init', ki, ei, 'tab:blue', ':'))
    for label, axis, energy, color, style in cases:
        # Display 12 occupied and 12 unoccupied bands around the gap.
        ax.plot(axis, energy[:, nocc-12:nocc+12] - zero, color=color, ls=style, lw=1.2)
        ax.plot([], [], color=color, ls=style, label=label)
        gap = energy[:,nocc].min() - energy[:,nocc-1].max()
        print('%s gap along sampled Gamma-X path: %.6f eV' % (label,gap))
    ax.set(xticks=[k[0],k[-1]], xticklabels=[r'$\Gamma$', 'X'],
           xlim=(k[0],k[-1]), ylabel=r'$E-E_{\mathrm{VBM}}^{\mathrm{SCF}}$ (eV)')
    ax.axhline(0,color='0.6',lw=0.5)
    ax.legend()
    plt.show()
    return fig


def read_energy(directory):
    text = (Path(directory) / 'stdout.txt').read_text(errors='replace')
    values = re.findall(r'^siesta:\s+Total\s*=\s*([-+0-9.EeDd]+)', text, re.M)
    if not values:
        raise ValueError('Final siesta: Total energy not found in ' + str(directory))
    return float(values[-1].replace('D', 'E').replace('d', 'e'))


def read_forces(directory):
    path = Path(directory) / 'Molecular.FA'
    count = int(path.read_text().splitlines()[0])
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    assert data.shape == (count, 4) and np.isfinite(data).all()
    np.testing.assert_array_equal(data[:, 0], np.arange(1, count + 1))
    return data[:, 1:]


def compare_observables(initial, ml, scf):
    """Compare run directories in initial, ML, SCF order."""
    scf = Path(scf)
    ref = density(scf / 'Molecular.RHO')
    e_ref, f_ref = read_energy(scf), read_forces(scf)
    cases = [('rho_init', initial, Path(initial) / 'In.RHO'),
             ('rho_ML', ml, Path(ml) / 'In.RHO'),
             ('rho_SCF', scf, scf / 'Molecular.RHO')]
    columns = []
    for label, directory, rho in cases:
        grid = density(rho)
        assert grid.shape == ref.shape and np.allclose(grid.cell, ref.cell)
        forces = read_forces(directory)
        assert forces.shape == f_ref.shape
        energy = read_energy(directory)
        delta = forces - f_ref
        columns.append([
            'Converged SCF' if label == 'rho_SCF' else 'One-step non-SCF',
            f'{100 * np.abs(grid.grid-ref.grid).sum()/ref.grid.sum():.6f}',
            f'{energy:.8f}',
            f'{abs(energy-e_ref)/len(f_ref)*1000:.6f}',
            f'{np.sqrt(np.mean(delta**2)):.6f}',
            f'{np.linalg.norm(delta, axis=1).max():.6f}',
            f'{np.linalg.norm(forces, axis=1).max():.6f}',
        ])
    labels = ['Calculation', 'Input density error (%)', 'Total energy (eV)',
              'Absolute energy error (meV/atom)', 'Force component RMSE (eV/Å)',
              'Maximum force-vector error (eV/Å)', 'Maximum force magnitude (eV/Å)']
    lines = ['| Quantity | ρ_init | ρ_ML | ρ_SCF |', '|---|---:|---:|---:|']
    lines += ['| ' + name + ' | ' + ' | '.join(values) + ' |'
              for name, values in zip(labels, zip(*columns))]
    table = '\n'.join(lines)
    return table


def plot_forces(initial, ml, scf):
    """Plot run directories in initial, ML, SCF order."""
    ref = read_forces(scf)
    cases = [('ρ_init', read_forces(initial), 'tab:blue', ':'),
             ('ρ_ML', read_forces(ml), 'tab:red', '--'),
             ('ρ_SCF', ref, 'black', '-')]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    atoms = np.arange(1, len(ref) + 1)
    for label, forces, color, style in cases:
        axes[0].plot(atoms, np.linalg.norm(forces, axis=1), style, color=color,
                     marker='.', label=label)
        axes[1].plot(atoms, np.linalg.norm(forces-ref, axis=1), style, color=color,
                     marker='.', label=label)
    axes[0].set(ylabel='Force magnitude (eV/Å)')
    axes[1].set(xlabel='Atom index', ylabel='Force-vector error vs SCF (eV/Å)')
    for ax in axes:
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.legend()
        ax.set_xlim(1, len(ref))
    plt.show()
    return fig


def plot_density_slice(ml, scf, y=20.0):
    """Compare one x-z slice; report the relative L1 error on the full 3D grid."""
    ref, predicted = density(scf), density(ml)
    assert predicted.shape == ref.shape and np.allclose(predicted.cell, ref.cell)
    lengths = np.diag(ref.cell)
    assert np.allclose(ref.cell, np.diag(lengths))
    if not 0 <= y < lengths[1]:
        raise ValueError('y must lie inside the cell.')
    index = min(int(np.rint(y / lengths[1] * ref.shape[1])), ref.shape[1] - 1)
    position = index * lengths[1] / ref.shape[1]
    slices = [predicted.grid[:, index, :], ref.grid[:, index, :]]
    vmax = max(values.max() for values in slices)
    x = np.arange(ref.shape[0]) * lengths[0] / ref.shape[0]
    z = np.arange(ref.shape[2]) * lengths[2] / ref.shape[2]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, title, values in zip(axes, ('DeepSCF', 'SCF'), slices):
        im = ax.pcolormesh(x, z, values.T, shading='nearest', cmap='viridis', vmin=0, vmax=vmax)
        ax.set(xlabel='x (Å)', ylabel='z (Å)', title=title,
               xlim=(0, lengths[0]), ylim=(0, lengths[2]), aspect='equal')
    fig.suptitle(f'Electron density at y = {position:.2f} Å')
    fig.colorbar(im, ax=axes, label=r'$\rho$ (e / $\mathrm{\AA}^3$)', shrink=0.85)
    error = 100 * np.abs(predicted.grid - ref.grid).sum(dtype=np.float64) / ref.grid.sum(dtype=np.float64)
    print('DeepSCF density error (full 3D grid): %.6f %%' % error)
    plt.show()
    return fig


def plot_density_error(ml, scf, structure, isovalue=0.05):
    """Show signed density-error isosurfaces on the original 3D grid."""
    import sisl.viz

    diff, ref = density(ml), density(scf)
    assert diff.shape == ref.shape and np.allclose(diff.cell, ref.cell)
    if isovalue <= 0:
        raise ValueError('isovalue must be positive.')
    diff.grid -= ref.grid
    geometry = sisl.get_sile(str(structure)).read_geometry()
    diff.set_geometry(geometry)
    levels = [
        {'val': isovalue, 'color': 'orange', 'opacity': 0.6, 'name': 'DeepSCF > SCF'},
        {'val': -isovalue, 'color': 'royalblue', 'opacity': 0.6, 'name': 'DeepSCF < SCF'},
    ]
    levels = [level for level in levels if diff.grid.min() < level['val'] < diff.grid.max()]
    if levels:
        fig = diff.plot(axes='xyz', interp=(1, 1, 1), smooth=True, isos=levels,
                        plot_geom=True, show_cell='box', backend='plotly')
    else:
        print('No error isosurface at this threshold; decrease isovalue to inspect smaller errors.')
        fig = geometry.plot(axes='xyz', show_cell='box', backend='plotly')
    fig.update_layout(title='Density error: DeepSCF − SCF')
    fig.show()
    return fig


def plot_dos(ml, scf, label='dna_single', energy_range=(-8, 5), sigma=0.10):
    """Compare nonmagnetic DOS with k weights and a common SCF Fermi zero."""
    from sisl.physics.electron import DOS
    from sisl.physics.distribution import get_distribution

    reference = sisl.get_sile(str(Path(scf) / (label + '.EIG')))
    zero = reference.read_fermi_level()
    k_ref, w_ref = sisl.get_sile(str(Path(scf) / (label + '.KP'))).read_data()
    shape_ref = reference.read_data().shape
    energy = np.linspace(*energy_range, 1301)
    distribution = get_distribution('gaussian', smearing=sigma)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for title, directory, color, style in [('SCF', scf, 'black', '-'),
                                            ('DeepSCF single-shot', ml, 'tab:red', '--')]:
        eig_reader = sisl.get_sile(str(Path(directory) / (label + '.EIG')))
        # sisl shifts each EIG to its own Ef; restore before using the common zero.
        eigenvalues = eig_reader.read_data() + eig_reader.read_fermi_level() - zero
        k, weights = sisl.get_sile(str(Path(directory) / (label + '.KP'))).read_data()
        assert eigenvalues.shape == shape_ref and eigenvalues.shape[0] == 1
        assert np.isfinite(eigenvalues).all() and len(weights) == eigenvalues.shape[1]
        assert np.allclose(k, k_ref) and np.allclose(weights, w_ref)
        assert np.isclose(weights.sum(), 1) and np.all(weights >= 0)
        # Each nonmagnetic spatial orbital represents two spin states.
        values = 2 * sum(w * DOS(energy, eig, distribution=distribution)
                         for w, eig in zip(weights, eigenvalues[0]))
        ax.plot(energy, values, color=color, ls=style, label=title)
    ax.axvline(0, color='0.6', lw=0.5)
    ax.set(xlabel=r'$E-E_F^{\mathrm{SCF}}$ (eV)', ylabel='DOS (states/eV/cell)',
           xlim=energy_range)
    ax.legend()
    plt.show()
    return fig
