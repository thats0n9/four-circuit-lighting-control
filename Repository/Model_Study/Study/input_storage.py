"""Resolve shared input files and stage isolated numerical runs."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent


def storage_map():
    return json.loads((HERE / 'input_storage_map.json').read_text('utf-8'))


def resolve_input(relative):
    local = (HERE / relative).resolve()
    if not local.is_relative_to(HERE):
        raise ValueError('Logical input path escapes the study directory')
    record = storage_map().get(relative)
    source = local if local.is_file() or record is None else (HERE / record['canonical']).resolve()
    if not source.is_relative_to(HERE.parent.parent) or not source.is_file():
        raise ValueError('Input file is unavailable: ' + relative)
    if record is not None:
        if source.stat().st_size != record['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Shared input integrity failed: ' + relative)
    return source


def verify_shared_inputs():
    records = storage_map()
    for relative in records:
        resolve_input(relative)
    return dict(status='passed', shared_inputs_checked=len(records))


def materialise_inputs(destination):
    destination = Path(destination).resolve()
    if destination.is_relative_to(HERE):
        raise ValueError('Shared inputs may only be staged outside the archived study')
    for relative, record in storage_map().items():
        source = resolve_input(relative)
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Staged input path escapes the destination')
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Staged input differs: ' + relative)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
