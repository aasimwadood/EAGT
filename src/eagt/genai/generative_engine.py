import os
from typing import List, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

nltk.download("punkt", quiet=True)

@dataclass
class GenResponse:
    text: str
    retrieved: Optional[List[str]] = None


class GenerativeEngine:
    """
    A simple LLM wrapper with optional BM25 retrieval-augmentation.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 180,
        temperature: float = 0.8,
        top_p: float = 0.92,
        rag_corpus_dir: str = "./rag_corpus",
        use_rag: bool = True,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.rag_corpus_dir = rag_corpus_dir
        self.use_rag = use_rag

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,
        )

        # Build RAG index
        self.docs, self.bm25 = self._build_index()

    def _build_index(self):
        docs, tokenized = [], []
        if not os.path.isdir(self.rag_corpus_dir):
            return [], None
        for fname in os.listdir(self.rag_corpus_dir):
            if fname.endswith(".txt"):
                text = open(os.path.join(self.rag_corpus_dir, fname), "r", encoding="utf8").read()
                docs.append(text)
                tokenized.append(nltk.word_tokenize(text.lower()))
        if not docs:
            return [], None
        return docs, BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(nltk.word_tokenize(query.lower()))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_idx]

    def generate(self, question: str, style: str, pedagogy: str) -> GenResponse:
        retrieved = self.retrieve(question) if self.use_rag else []

        prompt = (
            f"You are an empathetic AI tutor. Adapt your response based on the student's affect.\n"
            f"Affective style: {style}\n"
            f"Pedagogical strategy: {pedagogy}\n"
            f"Student question: {question}\n"
        )
        if retrieved:
            prompt += "\nSupporting material:\n" + "\n".join(retrieved[:2]) + "\n"

        outputs = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        return GenResponse(text=outputs[0]["generated_text"], retrieved=retrieved)
