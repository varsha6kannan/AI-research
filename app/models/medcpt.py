from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attn_mask: [B, T]
    mask = attn_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


@dataclass(frozen=True)
class BiEncoder:
    tokenizer: any
    model: any

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        dev = next(self.model.parameters()).device
        out: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                h = self.model(**tok).last_hidden_state
                pooled = _mean_pool(h, tok["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
            out.extend(pooled.detach().cpu().numpy().astype(np.float32).tolist())
        return out


@dataclass(frozen=True)
class CrossEncoder:
    tokenizer: any
    model: any

    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        dev = next(self.model.parameters()).device
        scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            q = [p[0] for p in batch]
            d = [p[1] for p in batch]
            tok = self.tokenizer(
                q,
                d,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(dev)
            with torch.no_grad():
                logits = self.model(**tok).logits  # [B, 1] or [B, C]
                if logits.dim() == 2 and logits.size(-1) == 1:
                    s = logits.squeeze(-1)
                else:
                    # Use first logit as a relevance proxy if multi-class.
                    s = logits[:, 0]
            scores.extend(s.detach().cpu().numpy().astype(np.float32).tolist())
        return [float(x) for x in scores]


@lru_cache(maxsize=1)
def get_medcpt_query_encoder() -> BiEncoder:
    tok = AutoTokenizer.from_pretrained(settings.medcpt_query_encoder)
    model = AutoModel.from_pretrained(settings.medcpt_query_encoder).to(_device())
    model.eval()
    return BiEncoder(tokenizer=tok, model=model)


@lru_cache(maxsize=1)
def get_medcpt_article_encoder() -> BiEncoder:
    tok = AutoTokenizer.from_pretrained(settings.medcpt_article_encoder)
    model = AutoModel.from_pretrained(settings.medcpt_article_encoder).to(_device())
    model.eval()
    return BiEncoder(tokenizer=tok, model=model)


@lru_cache(maxsize=1)
def get_medcpt_cross_encoder() -> CrossEncoder:
    tok = AutoTokenizer.from_pretrained(settings.medcpt_cross_encoder)
    model = AutoModelForSequenceClassification.from_pretrained(settings.medcpt_cross_encoder).to(
        _device()
    )
    model.eval()
    return CrossEncoder(tokenizer=tok, model=model)

