import json
from pathlib import Path
import sys
import zipfile
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acfo_structure_replication_io import write,read,sha,inventory,verify_request,safe_name,jobs,zip_return,audit_return


def request(tmp_path):
    root=tmp_path/'request';root.mkdir()
    write(root/'validation_contracts/a.json',dict(cases=[{'rings':8},{'rings':64}]))
    write(root/'REQUEST.json',dict(seed_base=20260906,fresh_processes=5,systems={
        'slab':dict(device='cpu',script='scripts/a.py',contract='validation_contracts/a.json')}))
    (root/'scripts').mkdir();(root/'scripts/a.py').write_text('print(1)\n')
    write(root/'PACKAGE_MANIFEST.json',dict(files=inventory(root)))
    return root


def test_changed_or_shadow_source_is_rejected(tmp_path):
    root=request(tmp_path);verify_request(root)
    (root/'scripts/shadow.py').write_text('print(2)')
    with pytest.raises(ValueError,match='Unmanifested'):verify_request(root)
    (root/'scripts/shadow.py').unlink()
    (root/'scripts/a.py').write_text('print(3)')
    with pytest.raises(ValueError,match='drift'):verify_request(root)


@pytest.mark.parametrize('name',['../secret','/absolute','C:/file','a\\b','a/../../file'])
def test_archive_paths_cannot_escape(name):
    with pytest.raises(ValueError):safe_name(name)


def test_confirmation_keeps_five_fresh_tasks_and_largest_case(tmp_path):
    root=request(tmp_path);tasks=jobs(root,'cpu','confirm')
    assert len(tasks)==5
    assert len({j['task']['seed'] for j in tasks})==5
    assert all(j['task']['case']=={'rings':64} for j in tasks)
    assert len(jobs(root,'cpu','smoke'))==1
    with pytest.raises(ValueError):jobs(root,'invalid','confirm')


def test_failed_worker_is_preserved_and_never_passes_numerical_gate(tmp_path):
    root=request(tmp_path);tasks=jobs(root,'cpu','smoke');out=tmp_path/'run';out.mkdir()
    write(out/'RUN.json',dict(profile='cpu',phase='smoke',request_manifest_sha256=sha(root/'PACKAGE_MANIFEST.json'),
        jobs={tasks[0]['id']:dict(status='failed')}))
    write(out/tasks[0]['id']/'task.json',tasks[0]['task'])
    archive=tmp_path/'return.zip';zip_return(out,archive)
    audit=audit_return(archive,root)
    assert audit['integrity_passed'] and not audit['numerical_passed']
    assert audit['failed_workers']==['slab_0'] and not audit['performance_eligible']
    # Truncation after transfer must fail inventory verification.
    truncated=tmp_path/'truncated.zip'
    with zipfile.ZipFile(archive) as source,zipfile.ZipFile(truncated,'w') as target:
        for n in source.namelist():
            if n!='slab_0/task.json':target.writestr(n,source.read(n))
    with pytest.raises(ValueError,match='inventory'):audit_return(truncated,root)


def test_missing_coverage_cannot_be_hidden_by_a_valid_inventory(tmp_path):
    root=request(tmp_path);out=tmp_path/'empty_run';out.mkdir()
    write(out/'RUN.json',dict(profile='cpu',phase='confirm',request_manifest_sha256=sha(root/'PACKAGE_MANIFEST.json'),jobs={}))
    archive=tmp_path/'empty.zip';zip_return(out,archive)
    with pytest.raises(ValueError,match='Coverage'):audit_return(archive,root)
