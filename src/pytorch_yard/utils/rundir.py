import os
from datetime import datetime
from pathlib import Path

import coolname
from dotenv import load_dotenv  # type: ignore

load_dotenv()


def setup_rundir():
    """
    Create a separate working directory under `$RESULTS_DIR/$WANDB_PROJECT` with a randomly generated run name.
    """
    date = datetime.now().strftime("%Y%m%d-%H%M")
    name = coolname.generate_slug(2)  # type: ignore
    os.environ['RUN_NAME'] = f'{date}-{name}'

    results_root = f'{os.getenv("RESULTS_DIR")}/{os.getenv("WANDB_PROJECT")}'
    if os.getenv('RUN_MODE', '').lower() == 'debug':
        run_dir = f'{results_root}/debug/{os.getenv("RUN_NAME")}'
        completed_run_dir = run_dir
        os.environ['WANDB_MODE'] = 'disabled'
    else:
        run_dir = f'{results_root}/running/{os.getenv("RUN_NAME")}'
        completed_run_dir = f'{results_root}/completed/{os.getenv("RUN_NAME")}'

    os.makedirs(run_dir, exist_ok=True)
    os.environ['RUN_DIR'] = run_dir
    os.environ['COMPLETED_RUN_DIR'] = completed_run_dir


def finish_rundir():
    """
    Move the run directory to completed subdirectory.
    """
    run_dir = Path(os.getenv('RUN_DIR', ''))
    completed_run_dir = Path(os.getenv('COMPLETED_RUN_DIR', ''))

    if run_dir != completed_run_dir:
        assert run_dir.exists() and run_dir.is_dir()
        assert not completed_run_dir.exists()
        os.makedirs(completed_run_dir.parent, exist_ok=True)
        run_dir.rename(completed_run_dir)
