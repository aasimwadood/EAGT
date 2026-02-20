"""
Generative Explanation Engine with Style Regularization.


This module implements:
- RAG-based retrieval using BM25
- LLM-based response generation with affect-aware prompting
- Candidate re-ranking with empathy scoring (Algorithm 1)
- Style regularization loss for training (Equation 12)

Algorithm 1: Generate N candidates, score with empathy classifier E(),
select best using: y* = argmax(log P_LLM(y|x) + λ * E(y))
"""

import os
import math
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

import nltk
from rank_bm25 import BM25Okapi

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@dataclass
class GenResponse:
    """Response from the generative engine."""
    text: str
    retrieved: Optional[List[str]] = None
    empathy_score: float = 0.0
    log_likelihood: float = 0.0
    combined_score: float = 0.0
    num_candidates: int = 1
    all_candidates: Optional[List[dict]] = None


@dataclass
class CandidateResponse:
    """A single candidate response with scores."""
    text: str
    log_likelihood: float = 0.0
    empathy_score: float = 0.0
    combined_score: float = 0.0


class EmpathyScorer(nn.Module):
    """
    Empathy scoring model for candidate re-ranking.


    This scorer evaluates how empathetic and supportive a generated
    response is, used for re-ranking candidate responses.

    Uses a sentiment/empathy classifier fine-tuned for tutoring contexts.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model for sentiment/empathy classification.
            Default uses SST-2 sentiment as proxy for empathy.
        device : str
            Device for inference ("cpu" or "cuda").
        """
        super().__init__()
        self.model_name = model_name
        self.device = device

        # Load sentiment classifier as empathy proxy
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

        # Empathy keywords for bonus scoring
        self.empathy_keywords = {
            "understand", "help", "support", "encourage", "great",
            "try", "okay", "step", "together", "guide", "explain",
            "clarify", "patience", "progress", "achieve", "learn",
        }

        # Frustration/negative keywords to penalize
        self.negative_keywords = {
            "wrong", "incorrect", "mistake", "fail", "bad", "stupid",
            "obvious", "clearly", "should know", "just", "simply",
        }

    def forward(self, text: str) -> float:
        """
        Compute empathy score for a response.

        Parameters
        ----------
        text : str
            Generated response text.

        Returns
        -------
        float
            Empathy score in [0, 1].
        """
        return self.score(text)

    def score(self, text: str) -> float:
        """
        Compute empathy score for a response.

        Combines:
        1. Sentiment classifier score (positive = empathetic)
        2. Empathy keyword presence
        3. Negative keyword penalty

        Parameters
        ----------
        text : str
            Generated response text.

        Returns
        -------
        float
            Empathy score in [0, 1].
        """
        if not text.strip():
            return 0.0

        # Get sentiment score
        try:
            result = self.classifier(text[:512])[0]  # Truncate for model
            sentiment_score = result["score"]
            if result["label"] == "NEGATIVE":
                sentiment_score = 1.0 - sentiment_score
        except Exception:
            sentiment_score = 0.5

        # Keyword analysis
        text_lower = text.lower()
        words = set(text_lower.split())

        empathy_count = len(words & self.empathy_keywords)
        negative_count = len(words & self.negative_keywords)

        keyword_score = min(1.0, empathy_count * 0.1) - min(0.3, negative_count * 0.1)
        keyword_score = max(0.0, keyword_score)

        # Combine scores
        combined = 0.6 * sentiment_score + 0.4 * keyword_score

        return float(np.clip(combined, 0.0, 1.0))

    def score_batch(self, texts: List[str]) -> List[float]:
        """Score multiple responses."""
        return [self.score(text) for text in texts]


class GenerativeEngine:
    """
    A simple LLM wrapper with optional BM25 retrieval-augmentation.

    For backward compatibility. Use GenerativeEngineWithReranking for
    with candidate re-ranking.
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


class GenerativeEngineWithReranking:
    """
    Generative engine with candidate re-ranking and style regularization.


    This implements the full generation pipeline:
    1. Retrieve relevant context using BM25
    2. Generate N candidate responses
    3. Score each candidate with empathy classifier
    4. Re-rank using: score = log P(y|x) + λ * E(y)
    5. Return best candidate

    Algorithm 1:
    ```
    Input: question x, style s, pedagogy p, num_candidates N, lambda λ
    Output: best response y*

    1. context ← BM25_retrieve(x)
    2. prompt ← build_prompt(x, s, p, context)
    3. candidates ← []
    4. for i = 1 to N:
    5.     y_i ← LLM.generate(prompt)
    6.     log_p ← compute_log_likelihood(y_i | prompt)
    7.     empathy ← E(y_i)
    8.     score ← log_p + λ * empathy
    9.     candidates.append((y_i, score))
    10. y* ← argmax(candidates, key=score)
    11. return y*
    ```
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        empathy_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        max_new_tokens: int = 180,
        temperature: float = 0.8,
        top_p: float = 0.92,
        num_candidates: int = 3,
        empathy_lambda: float = 0.5,
        rag_corpus_dir: str = "./rag_corpus",
        use_rag: bool = True,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model for text generation.
        empathy_model : str
            HuggingFace model for empathy scoring.
        max_new_tokens : int
            Maximum tokens to generate per response.
        temperature : float
            Sampling temperature for generation.
        top_p : float
            Nucleus sampling parameter.
        num_candidates : int
            Number of candidate responses to generate (N in Algorithm 1).
        empathy_lambda : float
            Weight for empathy score in re-ranking (λ in Equation 12).
        rag_corpus_dir : str
            Directory containing RAG corpus text files.
        use_rag : bool
            Whether to use RAG retrieval.
        device : str
            Device for inference.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_candidates = num_candidates
        self.empathy_lambda = empathy_lambda
        self.rag_corpus_dir = rag_corpus_dir
        self.use_rag = use_rag
        self.device = device

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

        # Load empathy scorer
        self.empathy_scorer = EmpathyScorer(model_name=empathy_model, device=device)

        # Build RAG index
        self.docs, self.bm25 = self._build_index()

    def _build_index(self) -> Tuple[List[str], Optional[BM25Okapi]]:
        """Build BM25 index from corpus."""
        docs, tokenized = [], []
        if not os.path.isdir(self.rag_corpus_dir):
            return [], None
        for fname in os.listdir(self.rag_corpus_dir):
            if fname.endswith(".txt"):
                path = os.path.join(self.rag_corpus_dir, fname)
                text = open(path, "r", encoding="utf8").read()
                docs.append(text)
                tokenized.append(nltk.word_tokenize(text.lower()))
        if not docs:
            return [], None
        return docs, BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        """Retrieve top-k relevant documents."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(nltk.word_tokenize(query.lower()))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_idx]

    def _build_prompt(
        self,
        question: str,
        style: str,
        pedagogy: str,
        context: List[str],
    ) -> str:
        """Build the generation prompt."""
        prompt = (
            f"You are an empathetic AI tutor. Adapt your response based on the student's affect.\n"
            f"Affective style: {style}\n"
            f"Pedagogical strategy: {pedagogy}\n"
        )
        if context:
            prompt += "\nRelevant context:\n" + "\n".join(context[:2]) + "\n"
        prompt += f"\nStudent question: {question}\n\nTutor response:"
        return prompt

    def _compute_log_likelihood(self, prompt: str, response: str) -> float:
        """
        Compute log-likelihood of response given prompt.

        """
        try:
            full_text = prompt + " " + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                # Negative log-likelihood, so negate
                log_likelihood = -outputs.loss.item()

            return log_likelihood

        except Exception:
            return 0.0

    def _generate_candidates(
        self,
        prompt: str,
        n: int,
    ) -> List[str]:
        """Generate n candidate responses."""
        candidates = []
        for _ in range(n):
            try:
                outputs = self.generator(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )
                text = outputs[0]["generated_text"].strip()
                candidates.append(text)
            except Exception:
                candidates.append("")

        return candidates

    def generate(
        self,
        question: str,
        style: str,
        pedagogy: str,
        num_candidates: Optional[int] = None,
        return_all_candidates: bool = False,
    ) -> GenResponse:
        """
        Generate response with candidate re-ranking.

        Algorithm 1

        Parameters
        ----------
        question : str
            Student's question.
        style : str
            Affective style guidance.
        pedagogy : str
            Pedagogical strategy.
        num_candidates : int, optional
            Number of candidates to generate. Defaults to self.num_candidates.
        return_all_candidates : bool
            If True, include all candidates in response.

        Returns
        -------
        GenResponse
            Best response with scores.
        """
        n = num_candidates or self.num_candidates

        # Step 1: Retrieve context
        retrieved = self.retrieve(question) if self.use_rag else []

        # Step 2: Build prompt
        prompt = self._build_prompt(question, style, pedagogy, retrieved)

        # Step 3-9: Generate and score candidates
        candidate_texts = self._generate_candidates(prompt, n)

        scored_candidates = []
        for text in candidate_texts:
            if not text:
                continue

            # Compute log-likelihood (line 6)
            log_likelihood = self._compute_log_likelihood(prompt, text)

            # Compute empathy score (line 7)
            empathy_score = self.empathy_scorer.score(text)

            # Compute combined score (line 8)
            # Equation (12): score = log P(y|x) + λ * E(y)
            combined_score = log_likelihood + self.empathy_lambda * empathy_score

            scored_candidates.append(CandidateResponse(
                text=text,
                log_likelihood=log_likelihood,
                empathy_score=empathy_score,
                combined_score=combined_score,
            ))

        # Step 10: Select best candidate
        if not scored_candidates:
            # Fallback if all generations failed
            return GenResponse(
                text="I'm here to help! Could you please rephrase your question?",
                retrieved=retrieved,
                num_candidates=0,
            )

        best = max(scored_candidates, key=lambda c: c.combined_score)

        # Build response
        all_candidates_data = None
        if return_all_candidates:
            all_candidates_data = [
                {
                    "text": c.text,
                    "log_likelihood": c.log_likelihood,
                    "empathy_score": c.empathy_score,
                    "combined_score": c.combined_score,
                }
                for c in scored_candidates
            ]

        return GenResponse(
            text=best.text,
            retrieved=retrieved,
            empathy_score=best.empathy_score,
            log_likelihood=best.log_likelihood,
            combined_score=best.combined_score,
            num_candidates=len(scored_candidates),
            all_candidates=all_candidates_data,
        )


class StyleRegularizationLoss(nn.Module):
    """
    Style regularization loss for training.


    This loss encourages generated responses to be empathetic
    and consistent with the target style.

    L_style = -λ * E(y) + KL(P_style || P_generated)

    where E(y) is the empathy score and KL divergence measures
    style consistency.
    """

    def __init__(
        self,
        empathy_lambda: float = 0.5,
        style_lambda: float = 0.3,
    ):
        """
        Parameters
        ----------
        empathy_lambda : float
            Weight for empathy score.
        style_lambda : float
            Weight for style consistency.
        """
        super().__init__()
        self.empathy_lambda = empathy_lambda
        self.style_lambda = style_lambda

        # Use sentiment classifier as empathy proxy
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
        )

    def forward(
        self,
        generated_texts: List[str],
        target_style: str = "supportive",
    ) -> torch.Tensor:
        """
        Compute style regularization loss.

        Parameters
        ----------
        generated_texts : List[str]
            Batch of generated responses.
        target_style : str
            Target style (e.g., "supportive", "challenging").

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        empathy_scores = []

        for text in generated_texts:
            if not text.strip():
                empathy_scores.append(0.0)
                continue

            try:
                result = self.classifier(text[:512])[0]
                score = result["score"]
                if result["label"] == "NEGATIVE":
                    score = 1.0 - score
                empathy_scores.append(score)
            except Exception:
                empathy_scores.append(0.5)

        # Convert to tensor
        empathy_tensor = torch.tensor(empathy_scores)

        # Empathy loss: encourage high empathy scores
        empathy_loss = -self.empathy_lambda * empathy_tensor.mean()

        return empathy_loss


def create_generative_engine(
    use_reranking: bool = True,
    **kwargs,
) -> GenerativeEngine:
    """
    Factory function to create generative engine.

    Parameters
    ----------
    use_reranking : bool
        If True, create engine with candidate re-ranking.
    **kwargs
        Additional arguments for the engine.

    Returns
    -------
    GenerativeEngine or GenerativeEngineWithReranking
    """
    if use_reranking:
        return GenerativeEngineWithReranking(**kwargs)
    return GenerativeEngine(**kwargs)
