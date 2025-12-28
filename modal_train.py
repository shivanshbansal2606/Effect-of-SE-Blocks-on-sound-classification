import modal
import os
import subprocess
from pathlib import Path
from typing import Optional

app = modal.App("audio-cnn-config-trainer")

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "outputs",
    "wandb",
    "logs",
    "runs",
}

EXCLUDED_SUFFIXES = {".pt", ".pth", ".ckpt", ".log"}


def _should_ignore(path: Path) -> bool:
    try:
        rel_parts = path.relative_to(Path(".")).parts
    except ValueError:
        rel_parts = path.parts

    if any(part in EXCLUDED_DIRS for part in rel_parts):
        return True

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in EXCLUDED_SUFFIXES:
            return True
        if ".tfevents" in path.name:
            return True

    return False


# --- Image ---
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path="/root/audio-cnn",
        ignore=_should_ignore,
    )
)

# --- Volumes ---
data_vol = modal.Volume.from_name("esc50-data", create_if_missing=False)
out_vol = modal.Volume.from_name("audio-cnn-outputs", create_if_missing=False)


def _train_impl(cfg_file: str, wandb_key: Optional[str] = None):
    os.chdir("/root/audio-cnn")
    for subdir in ("models", "runs", "figures", "results"):
        Path(f"/outputs/{subdir}").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if wandb_key:
        env["WANDB_API_KEY"] = wandb_key
    subprocess.run(["python", "main.py", "--config", cfg_file], check=True, env=env)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/data": data_vol, "/outputs": out_vol},
    timeout=60 * 60 * 3,
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train(
    cfg_file: str = "configs/se_resnet.yaml",
    wandb_api_key: Optional[str] = None,
    wandb_entity: Optional[str] = None,
):
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    _train_impl(cfg_file, wandb_api_key or os.environ.get("WANDB_API_KEY"))


@app.local_entrypoint()
def main(cfg_file: str = "configs/se_resnet.yaml", wandb_key: Optional[str] = None):
    _train_impl(cfg_file, wandb_key)
