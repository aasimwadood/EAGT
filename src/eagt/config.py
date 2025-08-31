from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class DataConfig:
    dataset: str
    root: str
    split_csv: str
    face_fps: int = 8
    audio_sr: int = 16000
    window_sec: float = 1.0
    stride_sec: float = 0.5
    classes: List[str] = None

@dataclass
class PreprocessConfig:
    face_detector: str = "mtcnn"
    face_size: int = 112
    augment: bool = True

@dataclass
class ModelConfig:
    vision_backbone: str = "resnet18"
    audio_backbone: str = "wav2vec2"
    hidden_dim: int = 256
    lstm_layers: int = 2
    dropout: float = 0.3

@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    grad_clip: float = 1.0
    ckpt_dir: str = "checkpoints"
    log_interval: int = 10

@dataclass
class GenAIConfig:
    model_name: str = "gpt2"
    max_new_tokens: int = 180
    temperature: float = 0.8
    top_p: float = 0.92
    rag_corpus_dir: str = "./rag_corpus"
    use_rag: bool = True

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass
class Config:
    seed: int
    device: str
    data: DataConfig
    preprocess: PreprocessConfig
    model: ModelConfig
    train: TrainConfig
    genai: GenAIConfig
    server: ServerConfig

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return Config(
        seed=y["seed"],
        device=y["device"],
        data=DataConfig(**y["data"]),
        preprocess=PreprocessConfig(**y["preprocess"]),
        model=ModelConfig(**y["model"]),
        train=TrainConfig(**y["train"]),
        genai=GenAIConfig(**y["genai"]),
        server=ServerConfig(**y["server"]),
    )
