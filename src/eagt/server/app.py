from fastapi import FastAPI
from pydantic import BaseModel
from eagt.strategy.mapping import map_affect_to_strategy
from eagt.genai.generative_engine import GenerativeEngine
from eagt.config import load_config

app = FastAPI(title="EAGT API", version="1.0")

class TutorRequest(BaseModel):
    question: str
    affect_label: str  # one of: frustration | confusion | boredom | engagement

# Load config and initialize the generative engine once at startup
CFG_PATH = "configs/default.yaml"
cfg = load_config(CFG_PATH)

engine = GenerativeEngine(
    model_name=cfg.genai.model_name,
    max_new_tokens=cfg.genai.max_new_tokens,
    temperature=cfg.genai.temperature,
    top_p=cfg.genai.top_p,
    rag_corpus_dir=cfg.genai.rag_corpus_dir,
    use_rag=cfg.genai.use_rag,
)

@app.get("/")
def root():
    return {
        "name": "EAGT API",
        "status": "ok",
        "endpoints": ["/tutor (POST)"],
        "model": cfg.genai.model_name,
    }

@app.post("/tutor")
def tutor(req: TutorRequest):
    """Generate an affect-aware tutoring response."""
    strat = map_affect_to_strategy(req.affect_label)
    resp = engine.generate(req.question, strat.style, strat.pedagogy)
    return {
        "strategy": strat.name,
        "style": strat.style,
        "pedagogy": strat.pedagogy,
        "answer": resp.text,
        "retrieved_docs": resp.retrieved or [],
    }
