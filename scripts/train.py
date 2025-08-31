import argparse
from eagt.config import load_config
from eagt.training.train_affect import train_loop

def main():
    ap = argparse.ArgumentParser(description="Train EAGT multimodal affect model")
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/default.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_loop(cfg)

if __name__ == "__main__":
    main()
