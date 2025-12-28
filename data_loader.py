# -*- coding: utf-8 -*-
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import random
from torchvision import transforms
from functools import partial


# -----------------------------------------------------------
# Augmentations
# -----------------------------------------------------------

def time_mask(spec, T=40, num_masks=1):
    out = spec.clone()
    length = out.shape[2]
    for _ in range(num_masks):
        t = random.randint(0, T)
        t0 = random.randint(0, max(1, length - t))
        out[:, :, t0:t0+t] = 0
    return out

def freq_mask(spec, F=30, num_masks=1):
    out = spec.clone()
    mel_bins = out.shape[1]
    for _ in range(num_masks):
        f = random.randint(0, F)
        f0 = random.randint(0, max(1, mel_bins - f))
        out[:, f0:f0+f, :] = 0
    return out


def _apply_time_mask(spec, T, num_masks=1):
    return time_mask(spec, T=T, num_masks=num_masks)

def _apply_freq_mask(spec, F, num_masks=1):
    return freq_mask(spec, F=F, num_masks=num_masks)


# -----------------------------------------------------------
# Dataset
# -----------------------------------------------------------

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate, n_mels, n_fft, hop_length, duration, transform=None):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.transform = transform

        self.target_samples = sample_rate * duration

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        rel_path = self.df.iloc[idx]["filename"]  
        label = int(self.df.iloc[idx]["class_id"])

        audio_path = os.path.join(self.audio_dir, "audio", rel_path)

        signal, sr = librosa.load(audio_path, sr=self.sample_rate)

        if len(signal) < self.target_samples:
            pad = self.target_samples - len(signal)
            signal = np.pad(signal, (0, pad))
        else:
            signal = signal[:self.target_samples]

        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        mel_db = np.expand_dims(mel_db, 0)
        spec = torch.tensor(mel_db, dtype=torch.float32)

        if self.transform:
            spec = self.transform(spec)

        return spec, torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------
# Loader builder
# -----------------------------------------------------------

def get_data_loaders(config, train_csv, val_csv, test_csv=None):

        train_t = transforms.Compose([
            transforms.Lambda(
                partial(_apply_time_mask, T=config.training.time_mask_param)
            ),
            transforms.Lambda(
                partial(_apply_freq_mask, F=config.training.freq_mask_param)
            ),
        ])

        val_t = None

        train_set = AudioDataset(
            csv_path=train_csv,
            audio_dir=config.data.data_path,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            duration=config.data.duration,
            transform=train_t
        )

        val_set = AudioDataset(
            csv_path=val_csv,
            audio_dir=config.data.data_path,
            sample_rate=config.data.sample_rate,
            n_mels=config.data.n_mels,
            n_fft=config.data.n_fft,
            hop_length=config.data.hop_length,
            duration=config.data.duration,
            transform=val_t
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers
        )

        val_loader = DataLoader(
            val_set,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )

        if test_csv:
            test_set = AudioDataset(
                csv_path=test_csv,
                audio_dir=config.data.data_path,
                sample_rate=config.data.sample_rate,
                n_mels=config.data.n_mels,
                n_fft=config.data.n_fft,
                hop_length=config.data.hop_length,
                duration=config.data.duration,
                transform=val_t
            )

            test_loader = DataLoader(
                test_set,
                batch_size=config.data.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers
            )

            return train_loader, val_loader, test_loader

        return train_loader, val_loader
