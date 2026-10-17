"""Check the auxiliary residual-margin comparator reported in section S4.

This uses the stored spatial calibration and the auxiliary component grid.
It does not replace the 69-point protection or 208-point acceptance analysis.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def compare(actual, expected):
    assert len(actual) == len(expected)
    assert list(actual[0]) == list(expected[0])
    maximum = 0.0
    for a, e in zip(actual, expected):
        for key in a:
            try:
                av, ev = float(a[key]), float(e[key])
            except (ValueError, TypeError):
                assert str(a[key]) == str(e[key]), (key, a[key], e[key])
            else:
                assert np.isclose(av, ev, rtol=1e-12, atol=1e-12), (key, av, ev)
                maximum = max(maximum, abs(av - ev))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='An optional new directory for recalculated tables')
    args = parser.parse_args()
    study = Path(__file__).resolve().parent
    if not (study / 'spatial_calibration').is_dir():
        raise RuntimeError('Place this script at the Study root before running it.')
    calibration = {(r['case'], r['layout'], int(r['n_points']), r['method']): r
                   for r in rows(study / 'spatial_calibration/calibration_parameters.csv')}
    metrics = {(r['case'], r['layout'], int(r['n_points']), r['method']): r
               for r in rows(study / 'spatial_calibration/spatial_metrics.csv')}
    detail = []
    for scenario in rows(study / 'light_field_inputs/room_scenarios.csv'):
        case = scenario['case']
        with np.load(study / 'light_field_inputs' / (case + '_field.npz'), allow_pickle=False) as field, \
             np.load(study / 'spatial_calibration' / (case + '_predictions.npz'), allow_pickle=False) as prediction:
            assert field['final_day'].shape == (208,)
            for layout in ['depth', 'area', 'window']:
                for n in [8, 12, 16, 24]:
                    key = (case, layout, n, 'improved')
                    margin = float(calibration[key]['day_margin_lx'])
                    errors = np.array([200.0, 500.0])[:, None] * (
                        prediction[f'{layout}_{n}_improved_day'] - field['final_day'])[None, :]
                    assert errors.size == 416
                    detail.append(dict(case=case, layout=layout, n_points=n, day_margin_lx=margin,
                                       absolute_coverage_pct=float(100 * np.mean(abs(errors) <= margin)),
                                       positive_exceed_pct=float(100 * np.mean(errors > margin)),
                                       unsafe_zone_states=int(metrics[key]['unsafe_zone_states']),
                                       feasible_zone_states=int(metrics[key]['feasible_zone_states'])))
    summary = []
    for layout in ['depth', 'area', 'window']:
        for n in [8, 12, 16, 24]:
            selected = [r for r in detail if r['layout'] == layout and r['n_points'] == n]
            assert len(selected) == 36
            summary.append(dict(layout=layout, n_points=n, n_cases=36,
                                spatial_absolute_coverage_min_pct=min(r['absolute_coverage_pct'] for r in selected),
                                spatial_absolute_coverage_max_pct=max(r['absolute_coverage_pct'] for r in selected),
                                spatial_positive_exceed_max_pct=max(r['positive_exceed_pct'] for r in selected),
                                cases_positive_exceed=sum(r['positive_exceed_pct'] > 0 for r in selected),
                                unsafe_zone_states=sum(r['unsafe_zone_states'] for r in selected),
                                feasible_zone_states=sum(r['feasible_zone_states'] for r in selected)))
    outputs = {'all_configuration_daylight_coverage.csv': detail,
               'all_configuration_coverage_summary.csv': summary}
    differences = {name: compare(value, rows(study / 'auxiliary_diagnostics' / name))
                   for name, value in outputs.items()}
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError('An output directory must be new.')
        if destination.is_relative_to(study):
            raise ValueError('Keep new output outside the archived Study directory.')
        destination.mkdir(parents=True)
        for name, value in outputs.items():
            with (destination / name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(value[0]))
                writer.writeheader()
                writer.writerows(value)
    print(json.dumps({'status': 'passed', 'configuration_rows': len(detail),
                      'summary_rows': len(summary), 'component_values_per_configuration': 416,
                      'maximum_absolute_difference_by_file': differences}, indent=2))


if __name__ == '__main__':
    main()
