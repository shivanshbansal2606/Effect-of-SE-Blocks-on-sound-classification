import base64
import io
import logging
import os
import sys
import threading
import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, List, Any
from uuid import UUID, uuid4

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import AudioCNN
from models.baseline_model import baseline_resnet34
from models.se_resnet_model import se_resnet34


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "outputs" / "logs"
MODEL_DIR = BASE_DIR / "outputs" / "models"
DEFAULT_CONFIG = BASE_DIR / "configs" / "se_resnet.yaml"

DEFAULT_BASELINE_MODEL = MODEL_DIR / "best_model_baseline_resnet_acc.pth"
DEFAULT_SE_MODEL = MODEL_DIR / "best_model_se_resnet_acc.pth"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TrainRequest(BaseModel):
    """Payload for triggering a new training job."""

    config: str = Field(
        default=str(DEFAULT_CONFIG.relative_to(BASE_DIR)),
        description="Relative path to the OmegaConf YAML file.",
    )


class JobStatus(BaseModel):
    job_id: UUID
    status: str
    config: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_path: Optional[str] = None
    log_tail: Optional[List[str]] = None


class WeightArtifact(BaseModel):
    filename: str
    size_bytes: int
    updated_at: datetime


class InferenceRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded WAV audio.")
    top_k: int = Field(default=3, ge=1, le=10)
    model_name: str = Field(default="baseline_resnet", description="baseline_resnet | se_resnet")


ESC50_CLASS_NAMES = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
]


app = FastAPI(title="Audio CNN API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@dataclass
class ModelSpec:
    name: str
    env_var: str
    default_path: Path
    builder: Callable[[int], torch.nn.Module]
    forward_type: str  # "audio_cnn" | "resnet"


MODEL_SPECS: Dict[str, ModelSpec] = {
    "baseline_resnet": ModelSpec(
        name="baseline_resnet",
        env_var="BASELINE_MODEL_PATH",
        default_path=DEFAULT_BASELINE_MODEL,
        builder=baseline_resnet34,
        forward_type="resnet",
    ),
    "se_resnet": ModelSpec(
        name="se_resnet",
        env_var="SE_RESNET_MODEL_PATH",
        default_path=DEFAULT_SE_MODEL,
        builder=lambda num_classes: se_resnet34(num_classes=num_classes, reduction_ratio=32),
        forward_type="resnet",
    ),
}


def _downsample(values: np.ndarray, max_points: int = 4000) -> List[float]:
    if values.size <= max_points:
        return values.tolist()
    step = int(np.ceil(values.size / max_points))
    return values[::step].tolist()


class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.sample_rate = 22050
        self.duration = 5
        self.target_samples = self.sample_rate * self.duration
        self.mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=self.sample_rate // 2,
        )
        self.db_transform = T.AmplitudeToDB()
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_lock = threading.Lock()

    def _model_path(self, spec: ModelSpec) -> Path:
        override = os.environ.get(spec.env_var)
        return Path(override) if override else spec.default_path

    def _load_model(self, model_name: str):
        with self.model_lock:
            if model_name in self.models:
                return
            spec = MODEL_SPECS.get(model_name)
            if spec is None:
                raise HTTPException(status_code=400, detail=f"Unknown model {model_name}")
            path = self._model_path(spec)
            logger.info(f"Loading model {model_name} from {path}")
            if not path.is_file():
                error_msg = (
                    f"Checkpoint for {model_name} not found at {path}. "
                    "Train and save weights, or set the env var."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            try:
                checkpoint = torch.load(path, map_location=self.device)
                classes = checkpoint.get("classes", ESC50_CLASS_NAMES.copy())
                model = spec.builder(num_classes=len(classes))
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_name] = {
                    "model": model,
                    "classes": classes,
                    "spec": spec,
                }
                logger.info(f"Successfully loaded model {model_name} with {len(classes)} classes")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
                raise

    def _prepare_waveform(self, audio_bytes: bytes) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = T.Resample(sample_rate, self.sample_rate)(waveform)
        if waveform.size(1) < self.target_samples:
            pad = self.target_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, : self.target_samples]
        return waveform

    def _prepare_input(self, waveform: torch.Tensor) -> Dict[str, Any]:
        spec = self.mel(waveform)
        spec_db = self.db_transform(spec)
        spec_norm = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
        spec_tensor = spec_db.unsqueeze(0)
        return {
            "tensor": spec_tensor,
            "visual": spec_norm.squeeze(0).numpy(),
        }

    def _forward_audio_cnn(self, model: AudioCNN, x: torch.Tensor):
        feature_maps = {}
        outputs = model.conv1(x)
        feature_maps["conv1"] = outputs
        out = outputs
        for i, block in enumerate(model.layer1):
            out = block(out, feature_maps, prefix=f"layer1.block{i}")
        feature_maps["layer1"] = out
        for i, block in enumerate(model.layer2):
            out = block(out, feature_maps, prefix=f"layer2.block{i}")
        feature_maps["layer2"] = out
        for i, block in enumerate(model.layer3):
            out = block(out, feature_maps, prefix=f"layer3.block{i}")
        feature_maps["layer3"] = out
        for i, block in enumerate(model.layer4):
            out = block(out, feature_maps, prefix=f"layer4.block{i}")
        feature_maps["layer4"] = out
        out = model.avgpool(out)
        out = out.view(out.size(0), -1)
        out = model.dropout(out)
        logits = model.fc(out)
        return logits, feature_maps

    def _forward_resnet(self, model: torch.nn.Module, x: torch.Tensor):
        feature_maps = {}
        out = torch.relu(model.bn1(model.conv1(x)))
        feature_maps["conv1"] = out
        out = model.layer1(out)
        feature_maps["layer1"] = out
        out = model.layer2(out)
        feature_maps["layer2"] = out
        out = model.layer3(out)
        feature_maps["layer3"] = out
        out = model.layer4(out)
        feature_maps["layer4"] = out
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        logits = model.linear(out)
        return logits, feature_maps

    def _run_forward(self, spec: ModelSpec, model, x: torch.Tensor):
        if spec.forward_type == "audio_cnn":
            return self._forward_audio_cnn(model, x)
        return self._forward_resnet(model, x)

    def predict(self, request: InferenceRequest) -> Dict[str, Any]:
        model_name = request.model_name
        audio_str = request.audio_data
        if "," in audio_str and "base64" in audio_str.split(",")[0]:
            audio_str = audio_str.split(",", 1)[1]
        try:
            audio_bytes = base64.b64decode(audio_str)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 audio payload") from exc

        self._load_model(model_name)
        entry = self.models[model_name]
        model = entry["model"]
        classes = entry["classes"]
        spec = entry["spec"]

        waveform = self._prepare_waveform(audio_bytes)
        spec_data = self._prepare_input(waveform)
        model_input = spec_data["tensor"]

        with torch.no_grad():
            logits, feature_maps = self._run_forward(spec, model, model_input)
            probs = torch.softmax(logits[0], dim=0)

        top_k = min(request.top_k, len(classes))
        confs, indices = torch.topk(probs, top_k)
        predictions = [
            {
                "class": classes[idx],
                "confidence": float(conf),
            }
            for conf, idx in zip(confs, indices)
        ]

        viz = {name: self._format_feature_map(tensor) for name, tensor in feature_maps.items()}

        waveform_np = waveform.squeeze(0).numpy()
        return {
            "model_name": model_name,
            "predictions": predictions,
            "input_spectrogram": {
                "shape": list(spec_data["visual"].shape),
                "values": spec_data["visual"].tolist(),
            },
            "waveform": {
                "values": _downsample(waveform_np, 5000),
                "sample_rate": self.sample_rate,
                "duration": waveform_np.size / self.sample_rate,
            },
            "visualization": viz,
        }

    def _format_feature_map(self, tensor: torch.Tensor) -> Dict[str, Any]:
        data = tensor.detach().cpu()
        if data.dim() == 4:
            data = data.mean(dim=1)
        array = data[0].numpy()
        return {"shape": list(array.shape), "values": array.tolist()}


_jobs: Dict[UUID, Dict] = {}
_lock = threading.Lock()
inference_engine = InferenceEngine()


def _iso_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_config(config_path: str) -> Path:
    candidate = (BASE_DIR / config_path).resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Config path must stay within repo.")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_path}")
    return candidate


def _tail(path: Path, lines: int = 40) -> List[str]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        buffer = handle.readlines()
    return [line.rstrip("\n") for line in buffer[-lines:]]


def _launch_training(job_id: UUID, config_path: Path) -> None:
    log_path = LOG_DIR / f"{job_id}.log"
    cmd = [sys.executable, "main.py", "--config", str(config_path)]

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)

            process.wait()
            exit_code = process.returncode
            status = "succeeded" if exit_code == 0 else "failed"
    except Exception as exc:  # pragma: no cover - defensive logging
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[ERROR] {exc}\n")
        status = "failed"
        exit_code = -1
    finally:
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job.update(
                    status=status,
                    finished_at=_iso_now(),
                    exit_code=exit_code,
                    log_path=str(log_path.relative_to(BASE_DIR)),
                )


def _serialize_job(job_id: UUID, job: Dict, include_logs: bool = False) -> JobStatus:
    log_tail = None
    log_path = job.get("log_path")
    if include_logs and log_path:
        log_tail = _tail(BASE_DIR / log_path, lines=40)

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        config=job["config"],
        started_at=job["started_at"],
        finished_at=job.get("finished_at"),
        exit_code=job.get("exit_code"),
        log_path=log_path,
        log_tail=log_tail,
    )


@app.get("/health")
def healthcheck():
    with _lock:
        active_jobs = sum(1 for job in _jobs.values() if job["status"] == "running")
    return {"status": "ok", "active_jobs": active_jobs}


@app.post("/train", response_model=JobStatus, status_code=202)
def start_training(payload: TrainRequest):
    config_path = _resolve_config(payload.config)
    job_id = uuid4()
    job_record = {
        "status": "running",
        "config": str(config_path.relative_to(BASE_DIR)),
        "started_at": _iso_now(),
        "log_path": None,
    }

    with _lock:
        _jobs[job_id] = job_record

    thread = threading.Thread(
        target=_launch_training, args=(job_id, config_path), daemon=True
    )
    thread.start()

    return _serialize_job(job_id, job_record)


@app.get("/train", response_model=List[JobStatus])
def list_jobs():
    with _lock:
        jobs_snapshot = list(_jobs.items())
    return [_serialize_job(job_id, job, include_logs=False) for job_id, job in jobs_snapshot]


@app.get("/train/{job_id}", response_model=JobStatus)
def get_job(job_id: UUID):
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
    return _serialize_job(job_id, job, include_logs=True)


@app.get("/weights", response_model=List[WeightArtifact])
def list_weights():
    artifacts: List[WeightArtifact] = []
    for path in sorted(MODEL_DIR.glob("*.pth")):
        stat = path.stat()
        artifacts.append(
            WeightArtifact(
                filename=path.name,
                size_bytes=stat.st_size,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return artifacts


@app.post("/inference")
def run_inference(request: InferenceRequest):
    try:
        return inference_engine.predict(request)
    except FileNotFoundError as exc:
        logger.error(f"FileNotFoundError: {exc}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error in inference: {exc}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {str(exc)}") from exc


@app.get("/")
def root():
    return {
        "service": "audio-cnn-backend",
        "message": "Use /train for jobs, /weights for checkpoints, /inference for predictions.",
    }

