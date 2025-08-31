import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

from eagt.data.datasets import EAGTDataset
from eagt.models.fusion import FusionClassifier


def _make_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Builds train/val dataloaders.
    For simplicity this demo uses the same CSV for both;
    in practice, pass separate split CSVs in your config.
    """
    ds_train = EAGTDataset(cfg.data.split_csv)
    ds_val = EAGTDataset(cfg.data.split_csv)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    return {"train": dl_train, "val": dl_val}


def train_loop(cfg) -> None:
    """
    Trains the FusionClassifier on (face,audio,behavior) sequences
    using cross-entropy over 4 affect labels.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dls = _make_dataloaders(cfg)

    model = FusionClassifier(num_classes=len(cfg.data.classes)).to(device)
    opt = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    crit = torch.nn.CrossEntropyLoss()

    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    global_step = 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        pbar = tqdm(dls["train"], desc=f"Epoch {epoch}/{cfg.train.epochs}")
        for batch in pbar:
            face = batch["face"].to(device)    # (B,T,Df)
            audio = batch["audio"].to(device)  # (B,T,Da)
            behav = batch["behav"].to(device)  # (B,T,Db)
            y = batch["label"].to(device)      # (B,)

            opt.zero_grad(set_to_none=True)
            logits = model(face, audio, behav)  # (B,C)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()

            global_step += 1
            if global_step % cfg.train.log_interval == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    acc = (preds == y).float().mean().item()
                pbar.set_postfix(loss=float(loss.item()), acc=round(acc, 3))

        # Save checkpoint per epoch
        ckpt_path = os.path.join(cfg.train.ckpt_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # Quick validation pass
        evaluate(cfg, ckpt_path, silent=True)


@torch.no_grad()
def evaluate(cfg, ckpt_path: str, silent: bool = False) -> None:
    """
    Loads a checkpoint and reports macro metrics on the (val) loader.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dls = _make_dataloaders(cfg)

    model = FusionClassifier(num_classes=len(cfg.data.classes))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    ys, yh = [], []
    pbar = tqdm(dls["val"], desc="Eval") if not silent else dls["val"]
    for batch in pbar:
        face = batch["face"].to(device)
        audio = batch["audio"].to(device)
        behav = batch["behav"].to(device)
        y = batch["label"].to(device)

        logits = model(face, audio, behav)
        preds = torch.argmax(logits, dim=-1)

        ys.extend(y.cpu().tolist())
        yh.extend(preds.cpu().tolist())

    report = classification_report(ys, yh, target_names=cfg.data.classes, digits=4)
    if not silent:
        print("\n" + report)
