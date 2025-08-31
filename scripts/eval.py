import argparse
from eagt.config import load_config
from eagt.training.train_affect import evaluate

def main():
    ap = argparse.ArgumentParser(description="Evaluate EAGT multimodal affect model")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path (e.g., checkpoints/model_epoch1.pt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.ckpt, silent=False)

if __name__ == "__main__":
    main()
