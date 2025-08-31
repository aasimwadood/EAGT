import argparse
import uvicorn
from eagt.config import load_config

def main():
    ap = argparse.ArgumentParser(description="Serve EAGT FastAPI")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    uvicorn.run("eagt.server.app:app", host=cfg.server.host, port=cfg.server.port, reload=False)

if __name__ == "__main__":
    main()
