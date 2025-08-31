#!/usr/bin/env python3
"""
Prepare dataset splits for EAGT (DAiSEE / SEMAINE) and (optionally) extract audio.

CSV schema expected by EAGTDataset:
    video_path,audio_path,behav_json,label

Examples
--------
DAiSEE (build CSV + extract audio to 16kHz mono):
    python scripts/prepare_data.py \
        --dataset daisee \
        --root /datasets/DAiSEE \
        --out configs/daisee_split.csv \
        --audio-out /datasets/DAiSEE/audio16k \
        --extract-audio

SEMAINE (build CSV; audio already present):
    python scripts/prepare_data.py \
        --dataset semaine \
        --root /datasets/SEMAINE \
        --out configs/semaine_split.csv

SEMAINE (re-extract audio to uniform 16kHz mono):
    python scripts/prepare_data.py \
        --dataset semaine \
        --root /datasets/SEMAINE \
        --out configs/semaine_split_uniform.csv \
        --audio-out /datasets/SEMAINE/audio16k_uniform \
        --extract-audio
"""

import os
import csv
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Tuple


# ----------------------------- ffmpeg helpers ----------------------------- #
def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def extract_audio_ffmpeg(
    in_video: Path,
    out_wav: Path,
    target_sr: int = 16000,
    overwrite: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Extract 16 kHz mono WAV audio from a video using ffmpeg.

    Returns True on success, False otherwise.
    """
    if out_wav.exists() and not overwrite:
        if verbose:
            print(f"[SKIP] Audio already exists: {out_wav}")
        return True

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(in_video),
        "-ac", "1",              # mono
        "-ar", str(target_sr),   # sampling rate
        "-vn",                   # audio only (no video)
        "-loglevel", "error",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True)
        if verbose:
            print(f"[OK] Extracted audio → {out_wav}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed for {in_video}: {e}", file=sys.stderr)
        return False


# ----------------------------- DAiSEE builder ----------------------------- #
def prepare_daisee(
    root: Path,
    out_csv: Path,
    audio_out_dir: Optional[Path] = None,
    extract_audio: bool = False,
    target_sr: int = 16000,
    overwrite_audio: bool = False,
    verbose: bool = True,
) -> None:
    """
    Prepare DAiSEE split CSV. If extract_audio=True, audio is extracted with ffmpeg.

    Expected structure (common distribution variants):
        root/
          Clips/
            Train/*.avi
            Validation/*.avi
            Test/*.avi
          Labels/
            Train/*.txt
            Validation/*.txt
            Test/*.txt

    Label files typically contain one of:
        boredom | engagement | confusion | frustration
    """
    clips = root / "Clips"
    labels = root / "Labels"
    if not clips.exists() or not labels.exists():
        raise FileNotFoundError(f"DAiSEE not found: expected {clips} and {labels}")

    if extract_audio and not have_ffmpeg():
        print("[WARN] ffmpeg not found; proceeding without audio extraction.", file=sys.stderr)
        extract_audio = False

    rows = []
    phases = ["Train", "Validation", "Test"]
    for phase in phases:
        vdir = clips / phase
        ldir = labels / phase
        if not vdir.exists():
            if verbose:
                print(f"[WARN] Missing videos for phase: {phase} at {vdir}")
            continue

        for vid in sorted(vdir.glob("*.avi")):
            label_path = ldir / f"{vid.stem}.txt"
            if not label_path.exists():
                if verbose:
                    print(f"[WARN] Missing label for {vid.name}")
                continue

            lbl = label_path.read_text(encoding="utf-8", errors="ignore").strip().lower()
            # Standardize to EAGT label set (lowercase)
            if lbl not in {"boredom", "engagement", "confusion", "frustration"}:
                if verbose:
                    print(f"[WARN] Unexpected label '{lbl}' for {vid.name}; keeping raw.")
            # Determine audio path
            audio_path = ""
            if extract_audio and audio_out_dir is not None:
                # keep phase-based mirroring under audio_out_dir
                out_wav = audio_out_dir / phase / f"{vid.stem}.wav"
                ok = extract_audio_ffmpeg(vid, out_wav, target_sr=target_sr, overwrite=overwrite_audio, verbose=verbose)
                if ok:
                    audio_path = str(out_wav)

            rows.append([str(vid), audio_path, "", lbl])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "audio_path", "behav_json", "label"])
        w.writerows(rows)

    print(f"[OK] DAiSEE CSV → {out_csv}  (n={len(rows)})")


# ----------------------------- SEMAINE builder ---------------------------- #
def find_first(glob_dir: Path, patterns: Tuple[str, ...]) -> Optional[Path]:
    for pat in patterns:
        hits = list(glob_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def prepare_semaine(
    root: Path,
    out_csv: Path,
    audio_out_dir: Optional[Path] = None,
    extract_audio: bool = False,
    target_sr: int = 16000,
    overwrite_audio: bool = False,
    verbose: bool = True,
) -> None:
    """
    Prepare SEMAINE split CSV. By default uses provided audio files.
    If extract_audio=True, will re-extract to uniform 16 kHz mono.

    Common structure:
        root/SEMAINE_dataset/<session_id>/{audio,video,labels}
    """
    sess_root = root / "SEMAINE_dataset"
    if not sess_root.exists():
        raise FileNotFoundError(f"SEMAINE not found: expected {sess_root}")

    if extract_audio and not have_ffmpeg():
        print("[WARN] ffmpeg not found; proceeding without audio extraction.", file=sys.stderr)
        extract_audio = False

    rows = []
    for session in sorted(sess_root.iterdir()):
        if not session.is_dir():
            continue

        vdir = session / "video"
        adir = session / "audio"
        ldir = session / "labels"

        vid = find_first(vdir, ("*.avi", "*.mp4", "*.mkv")) if vdir.exists() else None
        aud = find_first(adir, ("*.wav", "*.flac", "*.mp3")) if adir.exists() else None
        lbl = find_first(ldir, ("*.txt", "*.lab", "*.json")) if ldir.exists() else None

        if not vid or not lbl:
            if verbose:
                print(f"[WARN] Skipping session (missing video/label): {session.name}")
            continue

        # read label (simple single-label per session for this prep script)
        label_val = ""
        try:
            if lbl.suffix.lower() == ".json":
                jd = json.loads(lbl.read_text(encoding="utf-8", errors="ignore"))
                label_val = (jd.get("label") or "").strip().lower()
            else:
                label_val = lbl.read_text(encoding="utf-8", errors="ignore").strip().lower()
        except Exception:
            if verbose:
                print(f"[WARN] Could not parse label for {session.name}; setting empty.")
            label_val = ""

        # (re)extract audio if asked; otherwise use existing audio if present
        audio_path = ""
        if extract_audio and audio_out_dir is not None:
            out_wav = audio_out_dir / f"{session.name}.wav"
            ok = extract_audio_ffmpeg(vid, out_wav, target_sr=target_sr, overwrite=overwrite_audio, verbose=verbose)
            if ok:
                audio_path = str(out_wav)
        else:
            if aud:
                audio_path = str(aud)

        rows.append([str(vid), audio_path, "", label_val])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "audio_path", "behav_json", "label"])
        w.writerows(rows)

    print(f"[OK] SEMAINE CSV → {out_csv}  (n={len(rows)})")


# --------------------------------- CLI ----------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Prepare EAGT dataset CSVs (and optional audio extraction).")
    ap.add_argument("--dataset", choices=["daisee", "semaine"], required=True, help="Dataset to prepare")
    ap.add_argument("--root", type=str, required=True, help="Dataset root directory")
    ap.add_argument("--out", type=str, default="splits.csv", help="Output CSV path")
    ap.add_argument("--audio-out", type=str, default=None, help="Directory to save extracted audio WAVs")
    ap.add_argument("--extract-audio", action="store_true", help="Extract audio with ffmpeg (16kHz mono)")
    ap.add_argument("--overwrite-audio", action="store_true", help="Overwrite existing WAVs")
    ap.add_argument("--sr", type=int, default=16000, help="Target sampling rate for extraction")
    ap.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    audio_out = Path(args.audio_out).expanduser().resolve() if args.audio_out else None
    verbose = not args.quiet

    if args.dataset == "daisee":
        prepare_daisee(
            root=root,
            out_csv=out_csv,
            audio_out_dir=audio_out,
            extract_audio=args.extract_audio,
            target_sr=args.sr,
            overwrite_audio=args.overwrite_audio,
            verbose=verbose,
        )
    else:
        prepare_semaine(
            root=root,
            out_csv=out_csv,
            audio_out_dir=audio_out,
            extract_audio=args.extract_audio,
            target_sr=args.sr,
            overwrite_audio=args.overwrite_audio,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
