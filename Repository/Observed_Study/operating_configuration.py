"""Recover the documented P/FZ settings without replaying operating-day inputs."""
from pathlib import Path
import json
import numpy as np


BITS = np.array([8, 4, 2, 1])
EDGES = np.array([0.0, 120.0, 350.0])
RATED_POWER_W = np.array([150.0, 153.0, 154.0, 151.0])


def build(data, results):
    from observed_analysis import STATES, basis, save, selections, write
    repository = Path(__file__).resolve().parents[1]
    reference_path = repository / 'Model_Study/Study/control_calibration/R16_reference_calibration.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    with np.load(data / 'geometry.npz', allow_pickle=False) as archive:
        points = archive['points'].copy()
    with np.load(data / 'commissioning.npz', allow_pickle=False) as archive:
        night = archive['night'].copy()
        day = archive['day'].copy()
        reference_readings = archive['reference'].copy()
    prior, window = basis(points, points)
    zones = np.minimum((points[:, 1] / 2.05).astype(int), 3)
    parameters = json.loads((results / 'fitted_parameters.json').read_text(encoding='utf-8'))
    designs = selections(points)
    power = STATES @ RATED_POWER_W
    settings = []
    fixed_patterns = []
    for count in [8, 12, 24]:
        selected = designs['Depth-stratified', count]
        fitted = next(item for item in parameters if item['layout'] == 'Depth-stratified' and item['n_points'] == count and item['model'] == 'Improved model')
        coefficients = np.asarray(fitted['lamp_coefficients'])
        circuit = prior * coefficients[:, 0] + coefficients[:, 1]
        daylight = window * fitted['day_coefficients'][0] + fitted['day_coefficients'][1]
        prediction = STATES @ circuit.T
        night_margin = float(np.max(np.abs(night[2:, :, selected] - prediction[None, :, selected])) + 0.03 * circuit.sum(axis=1).max() + 1)
        predicted_day = reference_readings[:4, :, None] * daylight[selected]
        day_margin = float(np.quantile(np.abs(predicted_day - day[:4, :, selected]), 0.95) + 2)
        deficit = np.zeros((3, 16, 4))
        for bin_index, background in enumerate(EDGES):
            values = prediction + background * daylight
            margin = night_margin + (day_margin if bin_index else 0)
            for zone in range(4):
                region = values[:, zones == zone]
                deficit[bin_index, :, zone] = np.maximum.reduce([(500 - region.mean(axis=1) + margin) / 500, (300 - region.min(axis=1) + margin) / 300, np.zeros(16)])
        patterns = [int(min(range(16), key=lambda state: (deficit[0, state, zone], power[state], state))) for zone in range(4)]
        settings.append({'policy': 'P' + str(count), 'points': count, 'night_margin_lx': night_margin, 'positive_bin_extra_margin_lx': day_margin, 'single_zone_night_patterns': patterns})
        if count == 12:
            for zone, state in enumerate(patterns, 1):
                fixed_patterns.append({'zone': zone, 'request_bit': int(BITS[zone - 1]), 'fixed_state_code': state, 'binary_circuit_state': format(state, '04b')})
    configuration = {
        'evaluation_grid': 'All 81 candidate positions, equal point weights within each of four zones',
        'operating_targets': {'mean_lx': 500, 'minimum_lx': 300},
        'daylight_bin_lower_lx': EDGES.tolist(),
        'reference_margin_lx': reference['margin_lx'],
        'reference_mapping': 'Piecewise-linear inverse through gray_knots/lux_knots, subtract lamp_camera_equivalent_lx at captured state, then reference margin',
        'reference_calibration_file': reference_path.relative_to(repository).as_posix(),
        'reference_knots': len(reference['gray_knots']),
        'out_of_calibration_grayscale': 'Outside first-minus-2 to last-plus-2 gray knots: lowest daylight bin',
        'nominal_circuit_power_W': RATED_POWER_W.tolist(),
        'ranking': 'Maximum positive normalised mean/minimum shortfall, nominal circuit power, then state code',
        'manual_constraints': 'Manual and confirmation-hold masks: (state & forced_on) == forced_on and (state & forced_off) == 0; manual-off zones removed from automatic requests',
        'fixed_zone_combination': 'Bitwise OR of requested-zone night patterns, then forced-on/forced-off masks; no global repair',
        'policies': settings,
        'timing_s': {'controller': 0.2, 'reference': 1, 'vacancy': 60, 'minimum_dwell': 10, 'manual_lock': 2700},
        'confirmation_frames': 5,
        'source_functions': ['Model_Study/Study/spatial_fitting.py:fit', 'Model_Study/Study/spatial_fitting.py:fast_planner', 'Model_Study/Study/spatial_fitting.py:fast_engine', 'Model_Study/Study/reference_calibration.py:bins_table'],
        'scope': 'Fixed-input controller configuration; operating-day electricity uses the cumulative-register records',
    }
    save(results / 'operating_policy_configuration.json', configuration)
    write(results / 'operating_fixed_patterns.csv', fixed_patterns)
    return configuration
