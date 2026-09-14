"""Guard a training output directory, including workers whose launcher has exited."""
import fcntl
import json
import os
import time
from pathlib import Path


def acquire_run_lock(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    handle=open(out/'.training.lock','a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'Another trainer already owns {out}') from None
    handle.seek(0);handle.truncate()
    json.dump(dict(pid=os.getpid(),acquired_unix=time.time()),handle);handle.flush()
    return handle  # Keep this descriptor alive until rank 0 exits.


def live_training_processes(out,project=None,proc_root=Path('/proc')):
    out=Path(out).resolve();project=Path(project or Path(__file__).resolve().parent).resolve()
    found=[]
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            state=proc.joinpath('stat').read_text().split(') ',1)[1].split()[0]
            if state=='Z':continue
            argv=proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
            script=next((arg for arg in argv if Path(arg).name=='train_strong.py'),None)
            if not script:continue
            cwd=proc.joinpath('cwd').resolve()
            resolved=(cwd/script).resolve() if not Path(script).is_absolute() else Path(script).resolve()
            if resolved!=project/'train_strong.py':continue
            if '--out' in argv:
                value=Path(argv[argv.index('--out')+1])
                destination=value.resolve() if value.is_absolute() else (cwd/value).resolve()
            else:destination=project.parents[1]/'runs/prior96/strong_mouse96'
            if destination!=out:continue
            found.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,UnicodeError):continue
    return sorted(found)
