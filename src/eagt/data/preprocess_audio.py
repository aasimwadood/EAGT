import torch
import torchaudio
from typing import Literal, Tuple
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def load_mono_resampled(wav_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, convert to mono, and resample to target_sr.

    Returns
    -------
    wav : torch.Tensor
        Shape (1, T) float32 in [-1, 1].
    sr : int
        Sampling rate after resampling (== target_sr).
    """
    wav, sr = torchaudio.load(wav_path)         # (C, T)
    if wav.dtype != torch.float32:
        wav = wav.float() / (32768.0 if wav.dtype == torch.int16 else 1.0)
    # mono
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    # resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def extract_audio_feats(
    wav_path: str,
    sr: int = 16000,
    backend: Literal["wav2vec2", "mfcc"] = "wav2vec2",
) -> torch.Tensor:
    """
    Extract audio features for affect recognition.

    Parameters
    ----------
    wav_path : str
        Path to a WAV/MP3/FLAC file.
    sr : int
        Target sampling rate for loading/resampling.
    backend : {"wav2vec2","mfcc"}
        - "wav2vec2": contextual embeddings (base, ~768 dims per frame)
        - "mfcc": compact spectral features (~40 dims)

    Returns
    -------
    feats : torch.Tensor
        Time–feature matrix. Shape is approximately:
          - wav2vec2: (T_w2v, 768)
          - mfcc:     (T_mfcc, 40)
        Note: T differs by backend due to internal hop sizes.
    """
    wav, sr = load_mono_resampled(wav_path, target_sr=sr)

    if backend == "wav2vec2":
        # HuggingFace Wav2Vec2 Base (English); swap to multilingual as needed.
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        model.eval()

        with torch.no_grad():
            inputs = processor(
                wav.squeeze().numpy(),
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            # last_hidden_state: (B, T_frames, 768)
            out = model(inputs.input_values).last_hidden_state
        return out.squeeze(0)  # (T_frames, 768)

    # --- MFCC fallback (lightweight, language-agnostic) ---
    # Returns (n_mfcc, T) → transpose to (T, n_mfcc)
    n_mfcc = 40
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc)(wav)  # (1, n_mfcc, T)
    return mfcc.squeeze(0).transpose(0, 1).contiguous()  # (T, 40)
