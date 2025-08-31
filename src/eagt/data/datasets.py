import torch
from torch.utils.data import Dataset
import pandas as pd

class EAGTDataset(Dataset):
    """
    Windowed multimodal dataset for EAGT.

    Expected CSV format:
        video_path,audio_path,behav_json,label

    Returns dict:
        {
          "face": (T, Df),
          "audio": (T, Da),
          "behav": (T, Db),
          "label": int
        }

    Notes:
        - For demonstration, this dataset currently generates
          random tensors if no real features are provided.
        - Replace with actual precomputed features for training
          on DAiSEE, SEMAINE, or classroom data.
    """
    def __init__(self, csv_path, face_dim=512, audio_dim=768, behav_dim=16, seq_len=16):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception:
            # fallback: dummy dataset
            self.df = pd.DataFrame({"label": [0,1,2,3]*4})
        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.behav_dim = behav_dim
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        T = self.seq_len
        face_feats = torch.randn(T, self.face_dim).float()
        audio_feats = torch.randn(T, self.audio_dim).float()
        behav_feats = torch.randn(T, self.behav_dim).float()
        label = int(self.df.iloc[idx % len(self.df)]["label"])
        return {
            "face": face_feats,
            "audio": audio_feats,
            "behav": behav_feats,
            "label": torch.tensor(label)
        }
