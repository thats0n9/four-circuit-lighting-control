"""Compare scientific files without conflating floating-point and byte equality.

Integers, Boolean flags, strings, shapes and field names must match exactly.
Only floating-point values have a declared numerical tolerance. Differences in
the archived files are never corrected or rounded by this comparison.
"""
from pathlib import Path
import csv,hashlib,json,math,re

ATOL=1e-12
RTOL=1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Comparison:
    def __init__(self):
        self.compared_floats=0;self.different_floats=0;self.max_absolute_difference=0.;self.failures=[]

    def fail(self,where,message):
        if len(self.failures)<8:self.failures.append({'location':where,'reason':message})

    def floating(self,a,b,where):
        self.compared_floats+=1
        if a==b:return
        if math.isnan(a) and math.isnan(b):return
        self.different_floats+=1
        if math.isfinite(a) and math.isfinite(b):self.max_absolute_difference=max(self.max_absolute_difference,abs(a-b))
        if not math.isclose(a,b,abs_tol=ATOL,rel_tol=RTOL):self.fail(where,'Floating values exceed the declared tolerance')

    def json_value(self,a,b,where):
        if isinstance(a,bool) or isinstance(b,bool):
            if type(a) is not type(b) or a!=b:self.fail(where,'Boolean flags differ')
        elif isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):self.fail(where,'Object fields differ');return
            for key in a:self.json_value(a[key],b[key],where+'.'+key)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):self.fail(where,'List lengths differ');return
            for i,(x,y) in enumerate(zip(a,b)):self.json_value(x,y,where+f'[{i}]')
        elif isinstance(a,int) or isinstance(b,int):
            if type(a) is not type(b) or a!=b:self.fail(where,'Integer values or types differ')
        elif isinstance(a,float) and isinstance(b,float):
            self.floating(float(a),float(b),where)
        elif type(a) is not type(b) or a!=b:self.fail(where,'Categorical values or types differ')

    def csv_files(self,a,b):
        with a.open(encoding='utf-8-sig',newline='') as fa,b.open(encoding='utf-8-sig',newline='') as fb:
            ra=list(csv.reader(fa));rb=list(csv.reader(fb))
        if len(ra)!=len(rb) or not ra or ra[0]!=rb[0]:self.fail('CSV','Row count or columns differ');return
        for i,(x,y) in enumerate(zip(ra[1:],rb[1:]),1):
            if len(x)!=len(y):self.fail(f'row {i}','Column count differs');continue
            for j,(va,vb) in enumerate(zip(x,y)):
                if va==vb:continue
                where=f'row {i}, {ra[0][j]}'
                if re.fullmatch(r'[+-]?\d+',va) and re.fullmatch(r'[+-]?\d+',vb):
                    if int(va)!=int(vb):self.fail(where,'Integer values differ')
                    continue
                try:aa=float(va);bb=float(vb)
                except ValueError:self.fail(where,'Categorical strings differ');continue
                self.floating(aa,bb,where)

    def npz_files(self,a,b):
        import numpy as np
        with np.load(a,allow_pickle=False) as za,np.load(b,allow_pickle=False) as zb:
            if set(za.files)!=set(zb.files):self.fail('NPZ','Array names differ');return
            for key in za.files:
                aa=za[key];bb=zb[key]
                if aa.shape!=bb.shape or aa.dtype!=bb.dtype:self.fail(key,'Array shape or dtype differs');continue
                if aa.dtype.kind not in 'fc':
                    if not np.array_equal(aa,bb):self.fail(key,'Non-floating array values differ')
                    continue
                equal=aa==bb;nan=np.isnan(aa)&np.isnan(bb);different=~(equal|nan)
                self.compared_floats+=aa.size;self.different_floats+=int(different.sum())
                finite=different&np.isfinite(aa)&np.isfinite(bb)
                if finite.any():self.max_absolute_difference=max(self.max_absolute_difference,float(np.max(np.abs(aa[finite]-bb[finite]))))
                if not np.all(np.isclose(aa,bb,atol=ATOL,rtol=RTOL,equal_nan=True)):self.fail(key,'Floating array values exceed the declared tolerance')


def compare_archives(source,generated):
    source=Path(source);generated=Path(generated);rows=[]
    for name in ['analysis_results','camera_reference']:
        for a in sorted((source/name).iterdir()):
            if not a.is_file() or a.suffix not in ['.csv','.json','.npz']:continue
            relative=a.relative_to(source);b=generated/relative;c=Comparison();exact=b.is_file() and digest(a)==digest(b)
            if not b.is_file():c.fail(str(relative),'Generated file is missing')
            elif not exact:
                if a.suffix=='.csv':c.csv_files(a,b)
                elif a.suffix=='.json':c.json_value(json.loads(a.read_text('utf-8-sig')),json.loads(b.read_text('utf-8-sig')),'JSON')
                else:c.npz_files(a,b)
            rows.append({'file':relative.as_posix(),'byte_identical':exact,'numerically_equivalent':not c.failures,
                         'different_floating_values':c.different_floats,'max_absolute_floating_difference':c.max_absolute_difference,'failures':c.failures})
    return {'science_files':len(rows),'byte_identical':sum(x['byte_identical'] for x in rows),
            'numerically_equivalent':sum(x['numerically_equivalent'] for x in rows),
            'float_absolute_tolerance':ATOL,'float_relative_tolerance':RTOL,
            'nonfloating_comparison':'Exact integers, Boolean flags, categorical strings, field names, shapes and dtypes',
            'max_absolute_floating_difference':max(x['max_absolute_floating_difference'] for x in rows),
            'files':rows}
