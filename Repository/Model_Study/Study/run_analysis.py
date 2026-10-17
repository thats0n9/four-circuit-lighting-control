"""Reproduce the publication dataset in a fresh directory."""
from pathlib import Path
import sys,subprocess,argparse,shutil,json,hashlib
from export_data_schema import export
from input_storage import materialise_inputs
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--destination',type=Path,required=True);args=ap.parse_args()
 src=Path(__file__).resolve().parent;dest=args.destination.resolve()
 if dest.exists():raise SystemExit('Use a fresh destination directory')
 dest.mkdir(parents=True)
 for p in src.glob('*.py'):shutil.copy2(p,dest/p.name)
 shutil.copy2(src/'input_storage_map.json',dest/'input_storage_map.json')
 for name in ['fixed_inputs','spatial_calibration','light_field_inputs','control_calibration','metering_data']:
  if (src/name).exists():shutil.copytree(src/name,dest/name,ignore=shutil.ignore_patterns('__pycache__'))
  else:(dest/name).mkdir()
 materialise_inputs(dest)
 commands=[['camera_reference_model.py'],['main_analysis.py','--workers',str(args.workers)],['action_diagnostics.py'],['extended_evaluation.py'],['reference_diagnostics.py'],['manuscript_values.py'],['event_context_analysis.py'],['quality_analysis.py'],['validation_evidence.py'],['verify_analysis.py']]
 for command in commands:
  print('Running '+command[0],flush=True);subprocess.run([sys.executable,*command],cwd=dest,check=True)
 export(dest)
 subprocess.run([sys.executable,'verify_analysis.py'],cwd=dest,check=True)
 export(dest)
 comparisons=[]
 for root in ['analysis_results','camera_reference']:
  for q in sorted((src/root).iterdir()):
   if not q.is_file() or q.suffix not in ['.csv','.json','.npz']:continue
   p=dest/root/q.name;comparisons.append(dict(file=root+'/'+q.name,existed=p.exists(),byte_identical=p.exists() and sha(q)==sha(p)))
 report=dict(science_files=len(comparisons),byte_identical=sum(r['byte_identical'] for r in comparisons),mode='strict reproduction of the publication data schema',files=comparisons)
 (dest/'reproduction_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['science_files','byte_identical']}),flush=True)
 if report['science_files']!=report['byte_identical']:raise SystemExit('Inspect reproduction_audit.json')
if __name__=='__main__':main()
