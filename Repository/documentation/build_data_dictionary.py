"""Describe and verify fields in the primary observation inputs and results."""
from pathlib import Path
import argparse
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['file', 'field', 'unit', 'evidence_role', 'rows_or_shape']
TOKEN = re.compile(r'\[(\*|"(?:\\.|[^"\\])*")\]')


def records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def resolve_json_field(value, field):
    if not field.startswith('$'):
        raise ValueError('JSON fields must start with $')
    values = [value]
    position = 1
    for match in TOKEN.finditer(field):
        if match.start() != position:
            raise ValueError('Invalid JSON field path: ' + field)
        token = match.group(1)
        if token == '*':
            if not all(isinstance(item, list) for item in values):
                raise ValueError('Wildcard requires a list: ' + field)
            values = [child for item in values for child in item]
        else:
            key = json.loads(token)
            if not all(isinstance(item, dict) and key in item for item in values):
                raise ValueError('Missing JSON object field: ' + field)
            values = [item[key] for item in values]
        position = match.end()
    if position != len(field) or not values:
        raise ValueError('Unresolved JSON field: ' + field)
    return values


def units(field):
    name = field.lower()
    if 'lamp_coefficients' in name:
        return 'Columns: dimensionless circuit gain; lx offset'
    if 'day_coefficients' in name:
        return 'Model-dependent coefficients; see spatial-response equations'
    if 'uniformity' in name or 'fraction' in name or 'slack' in name:
        return 'dimensionless'
    if name.endswith('_pct') or '_pct"' in name or 'precision_pct' in name:
        return '%'
    if '_lxh' in name:
        return 'lx h'
    if 'kwh' in name:
        return 'kWh; group means and contrasts use kWh/day'
    if '_kw' in name:
        return 'kW'
    if name.endswith('_lx') or '_lx"' in name or 'rmse' in name or 'mae' in name:
        return 'lx'
    if field in ['night', 'day', 'reference', 'validation_readings', 'night_readings', 'day_readings', 'reference_readings', 'day_levels_lx']:
        return 'lx'
    if field == 'points' or name.endswith('_m'):
        return 'm'
    if 'hours' in name or name.endswith('_h'):
        return 'h; group mean durations use h/day'
    if name.endswith('_s') or '_s"' in name or name in ['event_seconds', 'sample_seconds']:
        return 's'
    return 'Identifier, category or quantity defined in the associated method'


def json_fields(value):
    grouped = {}
    def visit(item, field):
        if isinstance(item, dict) and item:
            for key, child in item.items():
                visit(child, field + '[' + json.dumps(key, ensure_ascii=False) + ']')
        elif isinstance(item, list) and item and all(isinstance(child, dict) for child in item):
            for child in item:
                visit(child, field + '[*]')
        else:
            grouped.setdefault(field, []).append(item)
    visit(value, '$')
    return grouped


def build_rows(root=ROOT):
    import numpy as np
    root = Path(root)
    result = []
    for directory, role in [('Observed_Study/data', 'Fixed observation input'), ('Observed_Study/results', 'Primary analysis or labelled diagnostic result')]:
        for path in sorted((root / directory).rglob('*')):
            if not path.is_file() or path.suffix not in {'.csv', '.json', '.npz'}:
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix == '.csv':
                values = records(path)
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    fields = next(csv.reader(stream))
                entries = [(field, str(len(values))) for field in fields]
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    entries = [(field, str(data[field].shape)) for field in data.files]
            else:
                groups = json_fields(json.loads(path.read_text('utf-8-sig')))
                entries = []
                for field, values in groups.items():
                    shapes = sorted({str(np.asarray(value).shape) for value in values})
                    entries.append((field, str(len(values)) + ' occurrence(s); value shape ' + ' or '.join(shapes)))
            result.extend(dict(file=relative, field=field, unit=units(field), evidence_role=role, rows_or_shape=shape) for field, shape in entries)
    return result


def verify_dictionary(root=ROOT, dictionary=None):
    import numpy as np
    root = Path(root)
    dictionary = records(root / 'documentation/primary_data_dictionary.csv') if dictionary is None else dictionary
    cache = {}
    for row in dictionary:
        path = (root / row['file']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Invalid dictionary source: ' + row['file'])
        if path not in cache:
            if path.suffix == '.json':
                cache[path] = json.loads(path.read_text('utf-8-sig'))
            elif path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    cache[path] = next(csv.reader(stream))
            else:
                with np.load(path, allow_pickle=False) as data:
                    cache[path] = data.files
        if path.suffix == '.json':
            resolve_json_field(cache[path], row['field'])
        elif row['field'] not in cache[path]:
            raise ValueError('Missing field: ' + row['file'] + ':' + row['field'])
    if len({(row['file'],row['field']) for row in dictionary}) != len(dictionary):
        raise ValueError('Duplicate file/field dictionary entry')
    return dict(status='passed', fields_checked=len(dictionary), files_checked=len(cache))


def write_dictionary(path, dictionary):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(dictionary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(ROOT.resolve()):
        raise ValueError('Use a new destination outside the archived repository')
    dictionary = build_rows()
    report = verify_dictionary(dictionary=dictionary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_dictionary(destination, dictionary)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
