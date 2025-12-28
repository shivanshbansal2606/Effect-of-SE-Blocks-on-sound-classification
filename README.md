Here’s a clean, professional, and visually engaging `README.md` tailored for your ESC-50 audio classification project:

```markdown
# 🎧 ESC-50 Audio Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-005571?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Logged-FFBE00?logo=weightsandbiases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A reproducible machine learning pipeline for environmental sound classification using the **ESC-50** dataset. This project implements audio preprocessing, ResNet-based CNN models, experiment tracking, and a user-friendly inference interface—all built with scientific rigor and MLOps best practices.

---

## 🌟 Features

- ✅ **Reproducible training**: Fixed random seeds, deterministic algorithms
- 🧠 **ResNet18-based audio classifier** trained on 50 sound classes
- 📊 **Experiment tracking** with Weights & Biases (`wandb`)
- 🚀 **Production-ready API** using FastAPI
- 🖥️ **Interactive demo UI** with Streamlit
- 📁 **Structured data pipeline** for ESC-50 (CSV + audio folders)
- 🧪 Support for **logistic regression baseline** and **CNN models**

---

## 📁 Project Structure

```bash
.
├── data/
│   └── ESC-50/                 # Processed dataset
│       ├── audio/              # .wav files
│       └── meta/               # esc50.csv metadata
├── models/                     # Trained model checkpoints
├── src/
│   ├── dataset.py              # Audio dataset & transforms
│   ├── model.py                # ResNet & logistic regression
│   ├── train.py                # Training loop with wandb logging
│   ├── api/                    # FastAPI inference server
│   └── app/                    # Streamlit GUI
├── notebooks/                  # EDA & prototyping
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/esc50-audio-cnn.git
cd esc50-audio-cnn
python -m venv venv && source venv/bin/activate  # or `.\venv\Scripts\activate` on Windows
pip install -r requirements.txt
```


### 2. Download & Prepare ESC-50

```bash
# Run the helper script to fetch and structure the dataset
python scripts/setup_esc50.py
```

> This creates the expected `data/ESC-50/` structure with properly formatted `esc50.csv`.

### 3. Train the Model

```bash
python src/train.py --epochs 50 --seed 42 --wandb_project esc50-study
```

> Uses Weights & Biases for logging. Set `WANDB_API_KEY` in your environment.

### 4. Launch Demo

```bash
# Start FastAPI backend (optional)
uvicorn src.api.main:app --reload

# OR run the Streamlit app (includes frontend + inference)
streamlit run src/app/app.py
```

Then open `http://localhost:8501` to classify your own audio clips!

---

## 📈 Results

| Model             | Accuracy (%) | Params (M) | Macro F1 Score | Percision | 
|-------------------|--------------|------------|----------------|-----------|
| SE-ResNet34       | 77.00        | ~22.0      | 76.30          | 76.42     |
| ResNet34 (Audio)  | 75.25        | ~21.8      | 74.24          | 74.31     |

*Trained on 5-fold cross-validation, ESC-50 test split*
NOTE: All these results are not under hard experimental conditions rather real world situations without any apparent hyperparameter tuning.

---

## 🔬 Reproducibility

All experiments are fully reproducible:
- Fixed `torch.manual_seed(42)`, `random.seed(42)`, `numpy.random.seed(42)`
- Deterministic CuDNN settings enabled
- All hyperparameters logged via `wandb`

---

## 📚 References

- [ESC-50 Dataset](https://github.com/karoldvl/ESC-50)
- Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*. ACM MM.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> Built by [Shivansh Bansal] 
> Final-year B.Tech 
