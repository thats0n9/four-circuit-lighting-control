"""Read fixed observation tables and prescribed trial inputs for analysis."""
import csv
import numpy as np
from input_storage import resolve_input


def arrays(name):
    with np.load(resolve_input('fixed_inputs/' + name), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def records(name):
    with resolve_input('fixed_inputs/' + name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def trial(record_id):
    data = arrays(f'input_{int(record_id)}.npz')
    data['seed'] = int(data['seed'])
    data['manuals'] = [tuple(int(value) for value in row) for row in data['manuals']]
    data['schedule'] = [tuple(int(value) for value in row) for row in data['schedule']]
    gray = np.clip(data['refgray'], 0, 254)
    fraction = (gray / 255) ** 2
    estimate = 1200 * fraction / (1 - fraction)
    data['bins'] = np.where(estimate >= 350, 2, np.where(estimate >= 120, 1, 0)).astype(np.int8)
    assert data['truth'].shape == (50400, 4)
    assert data['det'].shape == (252000, 4)
    assert data['valid'].shape == (252000,)
    assert data['window'].shape == data['refgray'].shape == (50400,)
    assert np.isfinite(data['window']).all() and np.isfinite(data['refgray']).all()
    return data
