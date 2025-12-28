# -*- coding: utf-8 -*-
# D:\KNOWLEDGE\Aminor\audio-cnn\main.py

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from io import BytesIO
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import librosa
import base64 # For handling base64 encoded audio if needed by frontend

# --- Define Modal App and Image ---
stub = modal.Stub("audio-cnn-inference")
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchaudio",
    "fastapi",
    "pydantic",
    "numpy",
    "librosa", # Add librosa for consistent preprocessing
    "modal-client" # Ensure modal client is available if needed within the container
)

# --- Define your Pydantic models for API requests/responses ---
class InferenceRequest(BaseModel):
    # Assuming the frontend sends audio as base64 encoded string
    audio: str

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float

# --- Model Architecture Definitions ---
# Include the exact model definitions used during training.
# This ensures the loaded state_dict matches the model structure.

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Basic Block (Used by ResNet-34) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- SE Basic Block (Used by SE-ResNet-34) ---
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, reduction_ratio=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction_ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) # Apply SE attention
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- ResNet with SE Blocks (General class that can take different block configs) ---
class SEResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, reduction_ratio=16):
        super(SEResNet, self).__init__()
        self.in_planes = 64
        self.reduction_ratio = reduction_ratio

        # Input convolution: expects 1 channel (Mel spectrogram) -> 64 channels
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Define the 4 main layers using the block type (SEBasicBlock for SE-ResNet)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Output layer: Global Average Pooling + Linear
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1) # First block of the layer might have stride, rest are 1
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.reduction_ratio))
            self.in_planes = planes * block.expansion # Update in_planes for the next block
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (batch_size, 1, n_mels, time_steps)
        out = F.relu(self.bn1(self.conv1(x))) # -> (batch_size, 64, n_mels, time_steps)

        out = self.layer1(out) # -> (batch_size, 64, n_mels/1, time_steps/1)
        out = self.layer2(out) # -> (batch_size, 128, n_mels/2, time_steps/2)
        out = self.layer3(out) # -> (batch_size, 256, n_mels/4, time_steps/4)
        out = self.layer4(out) # -> (batch_size, 512, n_mels/8, time_steps/8)

        # Global Average Pooling: (batch_size, 512, H, W) -> (batch_size, 512, 1, 1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # Reshape: (batch_size, 512, 1, 1) -> (batch_size, 512)
        out = out.view(out.size(0), -1)
        # Linear: (batch_size, 512) -> (batch_size, num_classes)
        out = self.linear(out)
        return out

# --- Helper Functions to Create Specific SE Architectures ---
def se_resnet34(num_classes=50, reduction_ratio=16):
    # Use [3, 4, 6, 3] blocks for ResNet-34 structure
    return SEResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, reduction_ratio=reduction_ratio)

# If you also trained a ResNet-34 baseline, you would define:
# def baseline_resnet34(num_classes=50):
#     return BaselineResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

# --- Modal Class for Model Loading and Prediction ---
@stub.cls(
    gpu="any", # Use any available GPU
    image=image,
    # Mount the directory containing your model checkpoint
    mounts=[modal.Mount.from_local_dir("./outputs", remote_path="/root/models")],
)
class Model:
    def __init__(self):
        print("Loading model...")
        # --- CRITICAL: Load your specific checkpoint ---
        # Update this path to point to your saved .pth file
        # Example: checkpoint_path = "/root/models/best_model_se_resnet34_acc.pth"
        # Example: checkpoint_path = "/root/models/best_model_se_resnet34_f1.pth"
        # Example: checkpoint_path = "/root/models/best_model_baseline_resnet34_acc.pth" (if using baseline)
        checkpoint_path = "/root/models/best_model_se_resnet34_acc.pth" # <-- CHANGE THIS TO YOUR CHECKPOINT NAME

        # --- CRITICAL: Define the model architecture matching your checkpoint ---
        # If using SE-ResNet-34:
        self.model = se_resnet34(num_classes=50, reduction_ratio=16) # Adjust num_classes and reduction_ratio if needed
        # If using ResNet-34 baseline, use the corresponding function/class definition
        # self.model = baseline_resnet34(num_classes=50) # Example placeholder

        # Load the state dictionary
        # Map to CPU first in case the model was trained on GPU, then it will be moved to GPU if available in the container
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        self.model.eval() # Set to evaluation mode
        print("Model loaded successfully.")

    # --- Preprocessing Function (MUST MATCH Training Preprocessing Exactly) ---
    def preprocess(self, audio_bytes: bytes):
        # Decode audio bytes using librosa
        signal, sr = librosa.load(BytesIO(audio_bytes), sr=22050, duration=5.0) # Match training config

        # Pad or trim to target length (5s * 22050 = 110250 samples)
        target_length = 22050 * 5
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
        else:
            signal = signal[:target_length]

        # Convert to Mel Spectrogram (Match training config)
        mel_spec = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=128, n_fft=2048, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize (Match training config - example: [0, 1])
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0) # Add channel dim -> (1, 128, ~216)

        # Convert to tensor
        spectrogram_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0) # Add batch dim -> (1, 1, 128, ~216)
        return spectrogram_tensor

    # --- Prediction Function ---
    @modal.method()
    def predict(self, audio_bytes: bytes):
        try:
            # Preprocess the audio
            spectrogram = self.preprocess(audio_bytes)

            # Run inference
            with torch.no_grad():
                output = self.model(spectrogram)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, dim=1)

                # --- CRITICAL: Map predicted index to class name ---
                # You need a list of class names corresponding to your model's output indices
                # This should match the order used during training (e.g., based on ESC-50 folder names or train.csv)
                # Example list for ESC-50 (replace with your actual class names in the correct order):
                class_names = [
                    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
                    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops",
                    "wind", "pouring_water", "toilet_flush", "thunderstorm", "crying_baby", "sneezing",
                    "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth",
                    "snoring", "drinking_sipping", "knock", "mouse_click", "keyboard_typing", "door_wood_knock",
                    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
                    "telephone", "door_wood_creaks", "glass_breaking", "helicopter", "chainsaw", "siren",
                    "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"
                ]

                predicted_class = class_names[predicted_idx.item()]
                confidence_score = confidence.item()

            return InferenceResponse(
                prediction=predicted_class,
                confidence=confidence_score
            )
        except Exception as e:
            print(f"Error during prediction: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

# --- FastAPI App ---
web_app = FastAPI()

@web_app.post("/predict", response_model=InferenceResponse)
async def predict_endpoint(request: InferenceRequest):
    # Decode base64 audio string from the request
    audio_bytes = base64.b64decode(request.audio)

    # Create a Modal client instance and call the predict method
    # The `@stub.cls` decorated `Model` class is instantiated and the `predict` method is called.
    # `call()` ensures synchronous execution within the FastAPI request handler.
    model_instance = Model() # This interacts with the deployed Modal class instance
    result = model_instance.predict.call(audio_bytes) # Use .call() for synchronous execution
    return result

# --- Modal ASGI App (for deployment) ---
@stub.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app

# --- Optional: Modal Local Entry Point (for testing the class directly if needed) ---
@stub.local_entrypoint()
def main():
    model_instance = Model()
    # Example: Load an audio file and predict (replace 'path/to/audio.wav' with an actual file path)
    # with open('path/to/audio.wav', 'rb') as f:
    #     audio_data = f.read()
    # result = model_instance.predict.call(audio_data)
    # print(result)