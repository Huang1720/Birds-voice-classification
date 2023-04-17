import os
import cv2
import librosa
import numpy as np
import soundfile as sf


def mono_to_color(X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    X = np.stack([X, X, X], axis=-1)
    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def trans_spec(path, period=5, img_size=224):
    for wav in os.listdir(path):
        signal, sr = sf.read(path + '\\' + wav)
        index = np.argmax(np.abs(signal))
        len_sig = len(signal)
        effective_length = sr * period
        if len_sig < effective_length:
            new_sig = np.zeros(effective_length, dtype=signal.dtype)
            start = np.random.randint(effective_length - len_sig)
            new_sig[start:start + len_sig] = signal
            signal = new_sig.astype(np.float32)
        elif len_sig > effective_length:
            start = int(index - period * sr / 2)
            if start < 0:
                start = 0
            elif start + effective_length > len_sig - 1:
                start = len_sig - 1 - effective_length
            else:
                pass
            signal = signal[start:start + effective_length].astype(np.float32)
        else:
            signal = signal.astype(np.float32)
        melspec = librosa.power_to_db(
            librosa.feature.melspectrogram(signal, sr=sr, n_mels=128, fmin=20, fmax=sr / 2)).astype(np.float32)
        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (img_size, img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
    return image
