import json
import subprocess
import sys
from pathlib import Path
from run_guard import acquire_run_lock,live_training_processes


def fake_process(root,pid,cwd,argv,state='S',ppid=1):
    proc=root/str(pid);proc.mkdir(parents=True)
    (proc/'stat').write_text(f'{pid} (python worker) {state} {ppid} 0 0 0\n')
    (proc/'cmdline').write_bytes(b'\0'.join(str(a).encode() for a in argv)+b'\0')
    (proc/'cwd').symlink_to(cwd,target_is_directory=True)


def test_orphan_worker_prevents_duplicate_resume(tmp_path):
    project=tmp_path/'project/scripts/prior96';project.mkdir(parents=True);proc=tmp_path/'proc';proc.mkdir()
    out=project.parents[1]/'runs/prior96/strong_mouse96'
    # The launcher no longer exists; its child has been reparented to PID 1.
    fake_process(proc,102,project,['python','-u','train_strong.py'],ppid=1)
    assert live_training_processes(out,project,proc)==[102]


def test_ignore_other_runs_and_zombies(tmp_path):
    project=tmp_path/'project/scripts/prior96';project.mkdir(parents=True);proc=tmp_path/'proc';proc.mkdir()
    out=project.parents[1]/'runs/prior96/strong_mouse96'
    fake_process(proc,101,project,['python','train_strong.py','--out','runs/other'])
    fake_process(proc,102,project,['python','train_strong.py'],state='Z')
    fake_process(proc,103,project,['python','/other/project/train_strong.py','--out',out])
    fake_process(proc,104,project,['python','-m','torch.distributed.run','train_strong.py','--out',out])
    assert live_training_processes(out,project,proc)==[104]


def test_directory_lock_excludes_other_writer_and_releases(tmp_path):
    out=tmp_path/'run';handle=acquire_run_lock(out)
    code='from run_guard import acquire_run_lock; import sys; h=acquire_run_lock(sys.argv[1])'
    args=[sys.executable,'-c',code,str(out)]
    first=subprocess.run(args,capture_output=True,text=True,cwd=Path(__file__).resolve().parent)
    assert first.returncode!=0 and 'Another trainer already owns' in first.stderr
    handle.close()
    second=subprocess.run(args,capture_output=True,text=True,cwd=Path(__file__).resolve().parent)
    assert second.returncode==0
