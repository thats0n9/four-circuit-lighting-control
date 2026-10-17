"""Supported commands for separate observation-led and numerical analyses."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def check_files():
    with (ROOT/'file_manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    for row in records:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size!=int(row['bytes']) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Archive integrity failed: '+row['path'])
    return dict(status='passed',files_checked=len(records))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['info','check-files','analyse','verify','draw','model-reproduce','summarise-model']);parser.add_argument('--destination',type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='info':print((ROOT/'release.json').read_text('utf-8'));return
    integrity=check_files()
    if args.command=='check-files':print(json.dumps(integrity,indent=2));return
    if args.command=='verify':
        subprocess.run([sys.executable,'-B',str(ROOT/'Observed_Study/verify_observations.py')],check=True);return
    if args.destination is None:raise ValueError('A fresh --destination is required')
    destination=args.destination.resolve()
    if destination.exists():raise ValueError('Existing destinations are never overwritten')
    if destination.is_relative_to(ROOT.resolve()):raise ValueError('Analysis destinations must be outside the archived repository')
    if args.command=='model-reproduce':
        command=[str(ROOT/'Model_Study/run_study.py'),'reproduce','--workers',str(args.workers),'--destination',str(destination)]
    elif args.command=='summarise-model':
        command=[str(ROOT/'Model_Study/summarise_model_mechanisms.py'),'--destination',str(destination)]
    else:
        name='observed_analysis.py' if args.command=='analyse' else 'draw_figures.py'
        command=[str(ROOT/'Observed_Study'/name),'--destination',str(destination)]
    subprocess.run([sys.executable,'-B',*command],check=True)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        print('ERROR: '+str(error),file=sys.stderr);raise SystemExit(1)
