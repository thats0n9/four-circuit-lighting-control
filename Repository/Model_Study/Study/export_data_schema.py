"""Serialise analysis outputs using the publication schema.

Fixed analysis inputs are never rewritten. Model, benchmark and sensitivity
descriptions are retained; output values are not filtered by wording.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ADMINISTRATIVE_FIELDS = {'provenance'}
RUN_REPORTS = {'reference_validation_summary.json', 'validation_evidence_summary.json'}


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if key not in ADMINISTRATIVE_FIELDS}
    if isinstance(value, list):
        return [clean(item) for item in value]
    return value


def export(root):
    root = Path(root).resolve()
    counts = {'csv': 0, 'json': 0, 'npz': 0, 'run_reports': 0}
    directories = [root / 'analysis_results', root / 'camera_reference', root / 'metering_data/metering_example/processed']
    for directory in directories:
        for path in sorted(directory.glob('*')):
            if not path.is_file():
                continue
            if path.name in RUN_REPORTS:
                destination = root / 'run_reports' / path.name
                destination.parent.mkdir(exist_ok=True)
                path.replace(destination)
                counts['run_reports'] += 1
                continue
            if path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    records = list(reader)
                if not fields:
                    continue
                fields = [key for key in fields if key not in ADMINISTRATIVE_FIELDS]
                with path.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows({key: record[key] for key in fields} for record in records)
                counts['csv'] += 1
            elif path.suffix == '.json':
                value = json.loads(path.read_text(encoding='utf-8-sig'))
                path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
                counts['json'] += 1
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in sorted(archive.files) if key not in ADMINISTRATIVE_FIELDS}
                np.savez_compressed(path, **arrays)
                counts['npz'] += 1
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(export(arguments.root), indent=2))
