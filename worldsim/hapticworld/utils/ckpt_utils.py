from pathlib import Path

import wandb


def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_latest_checkpoint(run_path: str, download_dir: Path) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    latest = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if latest is None or version_to_int(artifact) > version_to_int(latest):
            latest = artifact

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    latest.download(root=root)
    return root / "model.ckpt"


def load_run_config(path):
    """Load a run's .hydra/config.yaml, mapping class paths from the pre-release package name.

    Checkpoint configs written before the package was renamed reference
    ``interactive_world_sim.*`` in their ``_target_`` fields; the code now lives in ``hapticworld``.
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    text = OmegaConf.to_yaml(cfg, resolve=False)
    if "interactive_world_sim." in text:
        cfg = OmegaConf.create(text.replace("interactive_world_sim.", "hapticworld."))
    return cfg
