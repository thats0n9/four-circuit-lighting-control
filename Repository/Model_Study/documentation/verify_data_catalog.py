"""Check the numerical data catalog without changing data or results."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def glob_pattern(value):
    pattern = value.replace('Rxx', 'R[0-9][0-9]').replace('Dxx', 'D[0-9][0-9]')
    return re.sub(r'(?<=\d)x', '[0-9]', pattern)


def field_names(value, prefix=''):
    if isinstance(value, dict):
        fields = set()
        for key, child in value.items():
            fields.update(field_names(child, prefix + '.' + key if prefix else key))
        return fields
    if isinstance(value, list):
        if not value or all(not isinstance(child, (dict, list)) for child in value):
            return {prefix + '[]'}
        fields = set()
        for child in value:
            fields.update(field_names(child, prefix + '[]'))
        return fields
    return {prefix}


def catalog_summary(root=ROOT):
    formats = Counter(row['format'] for row in records(root / 'documentation/data_files.csv'))
    return dict(csv_data_files=formats['csv'], npz_data_files=formats['npz'],
                json_data_files=formats['json'], total_data_files=sum(formats.values()),
                dictionary_rows=len(records(root / 'documentation/data_dictionary.csv')),
                paper_output_entries=len(records(root / 'documentation/paper_output_map.csv')))


def verify_catalog(root=ROOT):
    root = Path(root).resolve()
    catalog = records(root / 'documentation/data_files.csv')
    dictionary = records(root / 'documentation/data_dictionary.csv')
    schemas = defaultdict(set)
    for row in dictionary:
        schemas[row['file_pattern']].add(row['field'])
    if len({row['path'] for row in catalog}) != len(catalog):
        raise ValueError('The catalog contains duplicate storage paths')
    for row in catalog:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.parent) or not path.is_file():
            raise ValueError('Invalid catalog path: ' + row['path'])
        if path.stat().st_size != int(row['bytes']) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Catalog file integrity failed: ' + row['path'])
        if row['format'] == 'csv':
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                fields = set(next(reader))
                count = sum(1 for item in reader)
        elif row['format'] == 'npz':
            with np.load(path, allow_pickle=False) as archive:
                fields = set(archive.files)
            count = len(fields)
        elif row['format'] == 'json':
            fields = field_names(json.loads(path.read_text('utf-8-sig')))
            count = len(fields)
        else:
            raise ValueError('Unsupported data format: ' + row['format'])
        if count != int(row['rows_or_array_keys_or_JSON_fields']):
            raise ValueError('Incorrect record or field count: ' + row['path'])
        if not fields.issubset(schemas[row['schema_pattern']]):
            raise ValueError('Undocumented data fields: ' + row['path'])
    for pattern in schemas:
        if not any(path.is_file() for path in root.glob(glob_pattern(pattern))):
            raise ValueError('Unresolved dictionary file family: ' + pattern)
    publications = records(root / 'documentation/paper_output_map.csv')
    for row in publications:
        matches = set()
        for pattern in row['repository_sources'].split('|'):
            selected = {path.resolve() for path in root.glob(glob_pattern(pattern)) if path.is_file()}
            if not selected or not all(path.is_relative_to(root.parent) for path in selected):
                raise ValueError('Invalid publication source: ' + pattern)
            matches.update(selected)
        if len(matches) != int(row['resolved_file_count']):
            raise ValueError('Incorrect publication source count: ' + row['caption'])
    summary = catalog_summary(root)
    if summary != json.loads((root / 'documentation/dictionary_coverage.json').read_text('utf-8-sig')):
        raise ValueError('Catalog summary counts do not match the current catalog')
    return dict(status='passed', data_files_verified=len(catalog),
                dictionary_rows_verified=len(dictionary), publication_entries_verified=len(publications),
                summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.report is not None:
        destination = args.report.resolve()
        if destination.exists() or destination.is_relative_to(ROOT.parent):
            parser.error('Choose a new report file outside Repository.')
    report = verify_catalog()
    if args.report is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
