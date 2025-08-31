# 🎓 Emotion-Aware Generative AI Tutor (EAGT)

**EAGT** is a research prototype that unifies **multimodal affect recognition** with **generative AI tutoring**.  
It detects learner states such as *frustration, confusion, boredom, and engagement* from video, audio, and behavioral cues, then adapts pedagogical strategies dynamically using an instruction-tuned LLM.

---

## ✨ Key Features
- **Multimodal Affective Computing**
  - Facial embeddings (CNN / ResNet18 backbone)
  - Speech embeddings (Wav2Vec2 / MFCC fallback)
  - Behavioral logs (keystrokes, clicks, timing)
  - Fusion with BiLSTM → 4 affect states

- **Adaptive Pedagogical Mapping**
  - Frustration → scaffold + encouragement  
  - Confusion → clarifying analogies + hints  
  - Boredom → gamified novelty  
  - Engagement → enrichment & challenge  

- **Generative AI Explanations**
  - Instruction-tuned LLM (default GPT-2, extendable to LLaMA/Mistral)  
  - Retrieval-Augmented Generation (BM25 over `rag_corpus/`)  
  - Style conditioning for empathetic tone  

- **Serving & Interaction**
  - REST API (FastAPI, port 8000)  
  - Interactive demo (Streamlit, port 8501)  

---

## 🗂️ Project Layout
src/eagt/
data/ # dataset loaders & preprocessing (video/audio)
models/ # vision, audio, and fusion models
training/ # training loop, evaluation
eval/ # metrics utilities
strategy/ # affect → pedagogy mapping
genai/ # generative engine (LLM + RAG)
server/ # FastAPI server
ui/
streamlit_app.py # browser demo
configs/
default.yaml # experiment config
rag_corpus/
note_*.txt # retrieval documents
scripts/
train.py, eval.py, serve.py


---

## ⚡ Quickstart

### Install locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

### Train affect model
```bash
python scripts/train.py --config configs/default.yaml
Evaluate checkpoint
python scripts/eval.py --config configs/default.yaml --ckpt checkpoints/model_epoch1.pt

Run API
python scripts/serve.py --config configs/default.yaml


→ Open http://localhost:8000/docs
```bash
Run Streamlit demo
streamlit run ui/streamlit_app.py


→ Open http://localhost:8501

🐳 Docker

Build and run API:
```bash
docker build -t eagt:latest .
docker run -p 8000:8000 eagt:latest


Compose (API + UI):
```bash
docker compose up --build


🔧 Usage

DAiSEE
```bash
python scripts/prepare_data.py --dataset daisee --root /datasets/DAiSEE --out configs/daisee_split.csv


SEMAINE
```bash
python scripts/prepare_data.py --dataset semaine --root /datasets/SEMAINE --out configs/semaine_split.csv

How to use

DAiSEE — build CSV + extract WAV audio
```bash
python scripts/prepare_data.py \
  --dataset daisee \
  --root /datasets/DAiSEE \
  --out configs/daisee_split.csv \
  --audio-out /datasets/DAiSEE/audio16k \
  --extract-audio


SEMAINE — build CSV (keep original audio)
```bash
python scripts/prepare_data.py \
  --dataset semaine \
  --root /datasets/SEMAINE \
  --out configs/semaine_split.csv


SEMAINE — re-extract uniform audio (optional)

```bash
python scripts/prepare_data.py \
  --dataset semaine \
  --root /datasets/SEMAINE \
  --out configs/semaine_split_uniform.csv \
  --audio-out /datasets/SEMAINE/audio16k_uniform \
  --extract-audio

📊 Evaluation

Affect recognition: accuracy, F1, confusion matrix

Learning outcomes: normalized gains

Engagement: self-report scales, persistence

Satisfaction: Likert ratings

See src/eagt/eval/metrics.py for metrics.

📖 References

Representative literature:

Corbett & Anderson (1994). Knowledge Tracing.

D’Mello & Kory (2015). Multimodal affect detection survey.

Whitehill et al. (2014). Engagement from facial cues.

Baltrušaitis et al. (2019). Multimodal fusion.

Graesser et al. (2012). AutoTutor.

Deng et al. (2024). ChatGPT in education.

Kasneci et al. (2023). Opportunities and risks of LLMs.


📜 License

Apache 2.0 — free for academic and non-commercial use. Contact authors for commercial licensing.

🙋 Contributing

PRs welcome! Add new modalities, integrate larger LLMs, or extend evaluation protocols.

