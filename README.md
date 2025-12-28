# Audio CNN

![alt text](thumbnail.png)

[Link to video](https://youtu.be/KLYfwigQPuY)

[Discord and more](https://www.andreastrolle.com/)

## Overview

Hi 🤙 In this video, you'll learn to train and deploy an audio classification CNN from scratch with PyTorch. I'll cover all the required concepts, so no prior experience is needed. The model will classify sounds like a dog barking or birds chirping from an audio file. You'll work with advanced techniques like Residual Networks (ResNet), data mixing, and Mel Spectrograms to build a robust training pipeline. Afterwards, we'll build a dashboard using Next.js and React to upload audio and visualize the model's internal layers to see what it "sees". The project uses Python, PyTorch, Next.js, React, and Tailwind, based on the T3 Stack. You can build along with me from start to finish. All services used are 100% free for you to use.

## Features:

- 🧠 Deep Audio CNN for sound classification
- 🧱 ResNet-style architecture with residual blocks
- 🎼 Mel Spectrogram audio-to-image conversion
- 🎛️ Data augmentation with Mixup & Time/Frequency Masking
- ⚡ Serverless GPU inference with Modal
- 📊 Interactive Next.js & React dashboard
- 👁️ Visualization of internal CNN feature maps
- 📈 Real-time audio classification with confidence scores
- 🌊 Waveform and Spectrogram visualization
- 🚀 FastAPI inference endpoint
- ⚙️ Optimized training with AdamW & OneCycleLR scheduler
- 📈 TensorBoard integration for training analysis
- 🛡️ Batch Normalization for stable & fast training
- 🎨 Modern UI with Tailwind CSS & Shadcn UI
- ✅ Pydantic data validation for robust API requests

## Setup

Follow these steps to install and set up the project.

### Clone the Repository

```bash
git clone https://github.com/Andreaswt/audio-cnn.git
```

### Install Python

Download and install Python if not already installed. Use the link below for guidance on installation:
[Python Download](https://www.python.org/downloads/)

Create a virtual environment with **Python 3.12**.

### Backend

Navigate to folder:

```bash
cd audio-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Modal setup:

```bash
modal setup
```

Run on Modal:

```bash
modal run main.py
```

Deploy backend:

```bash
modal deploy main.py
```

### Backend API service (FastAPI)

You can trigger training jobs *and* serve inference for the landing page through the FastAPI app.

```bash
cd audio-cnn
uvicorn api:app --host 0.0.0.0 --port 8000
```

Key endpoints:

- `POST /inference` – send `{ "audio_data": "<base64>", "top_k": 3, "model_name": "baseline_resnet" }`. `model_name` can be `baseline_resnet` or `se_resnet`.
- `POST /train` – start a config-driven training job.
- `GET /train`, `GET /train/{id}` – monitor queued jobs.
- `GET /weights` – list available `.pth` checkpoints.

Environment variables:

- `BASELINE_MODEL_PATH`, `SE_RESNET_MODEL_PATH` — point to the corresponding checkpoints (defaults to `outputs/models/best_model_*_acc.pth`).

### Frontend

Install dependencies:

```bash
cd audio-cnn-visualisation
npm i
```

Run:

```bash
npm run dev
```

Set `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`) so the upload flow can hit the FastAPI `/inference` endpoint. The UI now lets you toggle between Baseline ResNet and SE-ResNet before uploading; each call uses the selected backend checkpoint and renders its predictions, waveform, and feature maps.
