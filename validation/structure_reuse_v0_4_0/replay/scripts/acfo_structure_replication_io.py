"""Portable request/return integrity and frozen experiment coverage."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe relative path: {name}')
    return name


def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def verify_request(root):
    root=Path(root); manifest=read(root/'PACKAGE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        safe_name(name)
        if sha(root/name)!=digest:raise ValueError(f'Request drift: {name}')
    # Generated caches and compiled native binaries are allowed; extra Python
    # sources or contracts in import paths must never shadow the frozen code.
    for folder in ('scripts','src','validation_contracts'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.cpp','.h','.hpp'):
                if p.relative_to(root).as_posix() not in manifest['files']:
                    raise ValueError(f'Unmanifested executable/contract: {p}')
    return len(manifest['files'])


def jobs(root, profile, phase):
    if profile not in ('cpu','gpu','all') or phase not in ('smoke','confirm'):
        raise ValueError('Invalid profile or phase')
    spec=read(Path(root)/'REQUEST.json'); output=[]
    for name,system in spec['systems'].items():
        if profile!='all' and profile!=system['device']:continue
        design=read(Path(root)/system['contract'])
        cases=design.get('cases',design.get('sizes'))
        case=cases[0 if phase=='smoke' else -1]
        for i in range(1 if phase=='smoke' else spec['fresh_processes']):
            output.append(dict(id=f'{name}_{i}',system=name,script=system['script'],
                task={('size' if name=='odt' else 'case'):case,'seed':spec['seed_base']+100*i}))
    return output


def result_metrics(name, data, design, expected_task):
    """Recheck recorded numerical gates; performance is descriptive, not a gate."""
    def require(test,message):
        if not test:raise ValueError(f'{name}: {message}')
    def finite(value):
        if isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
        elif isinstance(value,(int,float)):require(math.isfinite(value),'nonfinite value')
    finite(data)
    require(data['seed']==expected_task['seed'],'seed mismatch')
    key='size' if name=='odt' else 'case'
    require(data[key]==expected_task[key],'case mismatch')
    ratios={}; values={}; errors=[]
    def method_set(m):require(set(m)==set(design['methods']),'missing or unexpected methods')
    def ratio(label,baseline,candidate):
        require(baseline>0 and candidate>0,'nonpositive timing')
        ratios[label]=baseline/candidate
    if name=='slab':
        methods=data['methods'];method_set(methods)
        require(len(data['audit'])==len(design['states']),'audit coverage')
        for a in data['audit']:
            require(max(a['reference_convergence'])<=design['reference_tolerance'],'reference convergence')
            require(a['independent_reference_errors'][0]<=design['operator_tolerance'],'physical field')
            require(max(a['independent_reference_errors'][1:])<=design['derivative_tolerance'],'derivatives')
            errors.extend(a['independent_reference_errors'])
        for m,v in methods.items():
            require(len(v['agreement_errors'])==len(design['states']),'state coverage')
            require(max(x for row in v['agreement_errors'] for x in row)<=1e-10,'method agreement')
            for n in design['calls_per_update']:
                for scope in ('warm_seconds','cold_seconds'):
                    t=v['totals'][str(n)][scope];require(t>0,'timing');values[f'{m}/{n}/{scope}']=t
        for n in design['calls_per_update']:
            for scope in ('warm_seconds','cold_seconds'):
                ratio(f'modal_over_shared/{n}/{scope}',methods['modal_cached']['totals'][str(n)][scope],methods['factor_shared']['totals'][str(n)][scope])
        ratio('separate_over_shared/apply',sum(methods['factor_separate']['apply_seconds']),sum(methods['factor_shared']['apply_seconds']))
    elif name=='relay4f':
        methods=data['methods'];method_set(methods)
        for m,v in methods.items():
            require(len(v['validation'])==len(design['phases']),'phase coverage')
            for a in v['validation']:
                require(a['physical_error']<=design['tolerance'],'physical error')
                require(a['quadrature_error']<=1e-10,'quadrature error')
                require(a['operator_relative_bound']<=design['tolerance'],'operator bound')
                errors.append(a['physical_error'])
            for rhs in design['rhs_counts']:
                for n in design['calls_per_mask']:
                    for scope in ('warm_seconds','cold_seconds'):
                        t=v['timings'][str(rhs)]['totals'][str(n)][scope]
                        require(t>0,'timing');values[f'{m}/{rhs}/{n}/{scope}']=t
        for rhs in design['rhs_counts']:
            for n in design['calls_per_mask']:
                for scope in ('warm_seconds','cold_seconds'):
                    for m in ('cpswf_core','nystrom_core','svd_core'):
                        ratio(f'dense_over_{m}/{rhs}/{n}/{scope}',values[f'dense_factor/{rhs}/{n}/{scope}'],values[f'{m}/{rhs}/{n}/{scope}'])
    elif name=='waxs':
        methods=data['methods'];method_set(methods)
        require(len(data['audit'])==len(design['states']),'audit coverage')
        for a in data['audit']:
            for k in ('direct_amplitude_relative','finite_difference_normalized'):
                limit=design['accuracy']['independent_amplitude_relative' if k.startswith('direct') else k]
                require(a[k]<=limit,k);errors.append(a[k])
        for m,v in methods.items():
            require(len(v['errors'])==len(design['states']),'state coverage')
            for e in v['errors']:
                for k in ('loss_absolute','gradient_relative'):require(e[k]<=design['accuracy'][k],k)
            require(all(len(s)==design['samples'] for s in v['sample_seconds']),'timing sample coverage')
            values[f'{m}/seconds']=v['three_state_seconds']
            values[f'{m}/peak_bytes']=max(v['incremental_torch_peak_bytes'])
        for base in ('production_backward','coordinate_fields'):
            for m in ('tangent_full','tangent_stream'):
                ratio(f'{base}_over_{m}/seconds',values[f'{base}/seconds'],values[f'{m}/seconds'])
    elif name=='odt':
        require(set(data['profiles'])==set(design['profiles']),'weight coverage')
        for profile,p in data['profiles'].items():
            require(p['energy_error']<=design['accuracy_tolerance'],'energy');errors.append(p['energy_error'])
            methods=p['methods'];method_set(methods)
            for m,v in methods.items():
                for k in ('agreement_relative','core_error','hermitian_error'):
                    require(v[k]<=design['accuracy_tolerance'],k);errors.append(v[k])
                require(len(v['normal_samples'])==design['samples'],'timing sample coverage')
                for k in ('core_seconds','normal_seconds','update_seconds','incremental_torch_peak_bytes'):
                    values[f'{profile}/{m}/{k}']=v[k]
                for n in design['calls_per_update']:values[f'{profile}/{m}/total_{n}']=v['totals'][str(n)]
            for m in ('fourier_sparse','view_update'):
                for k in ('core_seconds','normal_seconds'):
                    ratio(f'{profile}/fft_over_{m}/{k}',methods['fft_view'][k],methods[m][k])
    else:raise ValueError(name)
    return dict(ratios=ratios,values=values,max_recorded_error=max(errors))


def aggregate(rows):
    output={}
    for system in sorted(set(r['system'] for r in rows)):
        group=[r['metrics'] for r in rows if r['system']==system]
        output[system]={}
        for field in ('ratios','values'):
            output[system][field]={}
            for key in group[0][field]:
                samples=[g[field][key] for g in group]
                output[system][field][key]=dict(median=statistics.median(samples),min=min(samples),max=max(samples),samples=samples)
    return output


def zip_return(folder, archive):
    write(folder/'RETURN_MANIFEST.json',dict(files=inventory(folder)))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(folder).as_posix())
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')


def audit_return(archive,root):
    verify_request(root)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
        for n in names:safe_name(n)
        manifest=json.loads(z.read('RETURN_MANIFEST.json'))
        if set(names)!=set(manifest['files'])|{'RETURN_MANIFEST.json'}:raise ValueError('Return inventory mismatch')
        for n,h in manifest['files'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError(f'Return drift: {n}')
        run=json.loads(z.read('RUN.json'))
        if run['request_manifest_sha256']!=sha(Path(root)/'PACKAGE_MANIFEST.json'):raise ValueError('Wrong request')
        expected=jobs(root,run['profile'],run['phase'])
        if set(run['jobs'])!={j['id'] for j in expected}:raise ValueError('Coverage mismatch')
        checked=[];failed=[];spec=read(Path(root)/'REQUEST.json')
        for job in expected:
            ident=job['id'];record=run['jobs'][ident]
            task=json.loads(z.read(f'{ident}/task.json'))
            if task!=job['task']:raise ValueError(f'Task drift: {ident}')
            if record['status']!='passed':failed.append(ident);continue
            data=json.loads(z.read(f'{ident}/result.json'))
            contract=read(Path(root)/spec['systems'][job['system']]['contract'])
            checked.append(dict(system=job['system'],metrics=result_metrics(job['system'],data,contract,task)))
        return dict(integrity_passed=True,expected_workers=len(expected),passed_workers=len(checked),failed_workers=failed,
            numerical_passed=not failed,phase=run['phase'],profile=run['profile'],
            performance_eligible=run['phase']=='confirm' and not failed,
            performance_eligibility_scope='Complete frozen coverage only; idle logs, hardware and per-system paired results still require review.',
            summary=aggregate(checked) if checked else {})
