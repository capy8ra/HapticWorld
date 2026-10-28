"""Task registry for the HapticWorld release: box flip, whiteboard wipe, microwave door.

Everything task-specific that the scripts need is collected here, so a task is selected
with one flag (``--task wipe``) or one policy name (``wipe/it_sim``). Paths are relative
to the repository root, the folder that contains policy/, worldsim/, checkpoints/ and data/.

Policy names
  it_sim   image + torque ACT trained on demonstrations collected inside HapticWorld with force
           feedback on (the paper's "HapticWorld" column). One released checkpoint per task,
           checkpoints/policies/<task>/<task>_it_sim/ckpt_step_<N>/, N = the step the paper used.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

TASKS = {
    "boxflip": {
        "description": "push a box against a fixture and pivot it upright (gripper fixed)",
        "seeds": "data/seeds/boxflip/episode_*.hdf5",
        "demos": "data/demos/boxflip",
        # policy name -> (run directory under checkpoints/policies/<task>/, checkpoint subdir)
        "policies": {"it_sim": ("boxflip_it_sim", "ckpt_step_5106")},
        "crop": "center",       # camera frame center-cropped to a square
        "exec_steps": 5,        # actions executed per policy call inside the simulator (0.5 s)
        "steps": 185,           # policy steps per rollout; + 15 homing steps = 20 s at 10 Hz
        "train_stop_step": 5000,
    },
    "wipe": {
        "description": "erase marks from a whiteboard with a held eraser (gripper locked)",
        "seeds": "data/seeds/wipe/episode_*.hdf5",
        "demos": "data/demos/wipe",
        "policies": {"it_sim": ("wipe_it_sim", "ckpt_step_2000")},
        "crop": "center",       # camera frame center-cropped to a square
        "exec_steps": 5,
        "steps": 200,
        "train_stop_step": 2000,
    },
    "microwave": {
        "description": "grasp the door handle and pull the microwave door open (gripper live)",
        "seeds": "data/seeds/microwave/episode_*.hdf5",
        "demos": "data/demos/microwave",
        "policies": {"it_sim": ("microwave_it_sim", "ckpt_step_3000")},
        "crop": "right",        # the appliance and the arm sit in the right of the camera frame
        "exec_steps": 20,       # 2 s chunk renewal (the door swings; longer commitment works better)
        "steps": 200,
        "train_stop_step": 3000,
    },
}

# dataset_path values recorded in the released checkpoints' info.yaml (the folder names used
# during the paper experiments) -> the release layout under data/demos/
DEMO_ALIASES = {
    "sim_training": "boxflip",
    "wipe_sim_act": "wipe",
    "microwave_sim_act": "microwave",
}


def abs_path(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)


def wm_ckpt(task, stage):
    return abs_path(os.path.join("checkpoints", "wm", task, f"stage{stage}", "checkpoints", "last.ckpt"))


def seeds_glob(task):
    return abs_path(TASKS[task]["seeds"])


def demo_dir(dataset_path):
    """Map a dataset_path (as written in a checkpoint's info.yaml) to an existing folder.

    Accepts the release layout (absolute or relative to the repo root) and the folder
    names used during the paper experiments (DEMO_ALIASES)."""
    if dataset_path and os.path.isdir(dataset_path):
        return dataset_path
    norm = os.path.normpath(dataset_path or "")
    base = os.path.basename(norm)
    if base in DEMO_ALIASES:
        return abs_path(TASKS[DEMO_ALIASES[base]]["demos"])
    parts = norm.split(os.sep)
    if "demos" in parts:
        i = parts.index("demos")
        return abs_path(os.path.join("data", *parts[i:i + 2]))
    return abs_path(dataset_path)


def split_spec(spec):
    """'wipe/it_sim' -> ('wipe', 'it_sim'). Raises ValueError for anything else."""
    task, _, name = spec.partition("/")
    if task in TASKS and name and "/" not in name:
        return task, name
    raise ValueError(
        f"policy spec {spec!r} must be '<task>/<policy>' with task in {list(TASKS)} "
        f"(e.g. 'boxflip/it_sim') or a path to a model.pth")


def policy_path(task, name):
    """model.pth of a released policy; falls back to ckpt_latest when the preferred step is absent."""
    run, sub = TASKS[task]["policies"][name]
    root = abs_path(os.path.join("checkpoints", "policies", task, run))
    for cand in (sub, "ckpt_latest"):
        p = os.path.join(root, cand, "model.pth")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{task}/{name} -> {root} is not downloaded. Run: python checkpoints/download_assets.py --task {task}")
