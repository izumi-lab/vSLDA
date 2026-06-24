from __future__ import annotations

import importlib
import os
import re
import subprocess
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from src.core.artifacts import load_text_lines

_NUM_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_NEOLOGD_ALIASES = {
    "neologd",
    "ipadic-neologd",
    "mecab-ipadic-neologd",
}


def is_japanese_language(language: str | None) -> bool:
    normalized = (language or "").strip().lower()
    return normalized in {"ja", "japanese", "jp"}


def _clean_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).strip()


def _is_numeric_token(token: str) -> bool:
    return bool(_NUM_RE.fullmatch(token))


def _is_symbol_like_token(token: str) -> bool:
    cleaned = _clean_token(token)
    if not cleaned:
        return True
    return all(unicodedata.category(ch)[0] in {"P", "S"} for ch in cleaned)


@lru_cache(maxsize=4)
def _load_stopwords_file(path: str) -> frozenset[str]:
    values: set[str] = set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stopword file not found: {path}")
    for raw_line in load_text_lines(p):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values.add(_clean_token(line))
    return frozenset(values)


def load_japanese_stopwords(
    *,
    stopwords_path: str | None = None,
    extra_stopwords: Iterable[str] | None = None,
) -> set[str]:
    stopwords = set(_load_japanese_stopwords_library())
    if stopwords_path:
        stopwords.update(_load_stopwords_file(stopwords_path))
    if extra_stopwords:
        stopwords.update(_clean_token(w) for w in extra_stopwords if _clean_token(w))
    # Stopword filtering is disabled in tokenize_japanese_text, so returning an
    # empty set is valid and keeps callers backward-compatible.
    return stopwords


@lru_cache(maxsize=1)
def _load_japanese_stopwords_library() -> frozenset[str]:
    """
    Prefer a library-provided Japanese stopword list.
    Currently uses `stopwordsiso` when available.
    """
    try:
        import stopwordsiso as sw

        words = sw.stopwords("ja")
        if words:
            return frozenset(_clean_token(w) for w in words if _clean_token(w))
    except Exception:
        pass
    return frozenset()


@lru_cache(maxsize=2)
def _resolve_unidic_dir() -> str | None:
    for module_name in ("unidic", "unidic_lite"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        dicdir = getattr(module, "DICDIR", None)
        if dicdir:
            return str(dicdir)
    return None


@lru_cache(maxsize=1)
def _resolve_neologd_dir() -> str | None:
    env_candidates = [
        os.environ.get("MECAB_IPADIC_NEOLOGD_DIR"),
        os.environ.get("NEOLOGD_DICDIR"),
    ]
    for candidate in env_candidates:
        if candidate and Path(candidate).exists():
            return candidate

    path_candidates = [
        "/var/lib/mecab/dic/mecab-ipadic-neologd",
        "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd",
        "/usr/local/lib/mecab/dic/mecab-ipadic-neologd",
        "/usr/lib/mecab/dic/mecab-ipadic-neologd",
        "/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd",
        "/usr/local/libexec/mecab/dic/mecab-ipadic-neologd",
    ]
    for candidate in path_candidates:
        if Path(candidate).exists():
            return candidate

    try:
        proc = subprocess.run(
            ["mecab-config", "--dicdir"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            base = proc.stdout.strip()
            if base:
                candidate = str(Path(base) / "mecab-ipadic-neologd")
                if Path(candidate).exists():
                    return candidate
    except Exception:
        pass

    return None


def _resolve_dicdir_spec(dicdir: str | None) -> str | None:
    if not dicdir:
        return None
    normalized = dicdir.strip().lower()
    if normalized in _NEOLOGD_ALIASES:
        return _resolve_neologd_dir()
    return dicdir


@lru_cache(maxsize=8)
def _create_mecab_tagger(dicdir: str | None, require_unidic: bool):
    try:
        import MeCab
    except Exception as exc:
        raise RuntimeError(
            "MeCab is required for Japanese tokenization. "
            "Install mecab-python3 and a UniDic dictionary."
        ) from exc

    resolved_dicdir = _resolve_dicdir_spec(dicdir)
    if (
        resolved_dicdir is None
        and dicdir
        and dicdir.strip().lower() in _NEOLOGD_ALIASES
    ):
        raise RuntimeError(
            "mecab-ipadic-neologd dictionary was not found. "
            "Install it and/or set MECAB_IPADIC_NEOLOGD_DIR."
        )
    if resolved_dicdir is None:
        # Default preference: mecab-ipadic-neologd -> UniDic
        resolved_dicdir = _resolve_neologd_dir() or _resolve_unidic_dir()
    if require_unidic and not resolved_dicdir:
        raise RuntimeError(
            "No Japanese dictionary was found. "
            "Install mecab-ipadic-neologd or unidic (or unidic-lite), "
            "or pass ja_dicdir."
        )
    if resolved_dicdir and not Path(resolved_dicdir).exists():
        raise FileNotFoundError(f"MeCab dictionary path not found: {resolved_dicdir}")

    opts: list[str] = []
    if resolved_dicdir:
        opts.extend(["-d", resolved_dicdir])
    tagger = MeCab.Tagger(" ".join(opts))
    # Avoid occasional internal state issues in some MeCab builds.
    tagger.parse("")
    return tagger


def _feature_at(features: Sequence[str], idx: int) -> str:
    if 0 <= idx < len(features):
        return features[idx]
    return ""


def _select_base_form(surface: str, features: Sequence[str]) -> str:
    # Prefer lemma/base-form fields and avoid reading fields (e.g., Katakana).
    for idx in (10, 6):
        cand = _feature_at(features, idx)
        if cand and cand != "*":
            return cand
    return surface


def tokenize_japanese_text(
    text: str,
    *,
    replace_num: bool = True,
    stopwords: set[str] | None = None,
    dicdir: str | None = None,
    require_unidic: bool = True,
) -> list[str]:
    if not text or not text.strip():
        return []

    _ = stopwords  # retained for backward-compatible function signature
    tagger = _create_mecab_tagger(dicdir, require_unidic)

    tokens: list[str] = []
    node = tagger.parseToNode(text)
    while node is not None:
        surface = node.surface or ""
        if not surface:
            node = node.next
            continue

        features = (node.feature or "").split(",")
        token = _clean_token(_select_base_form(surface, features))
        if not token or token == "*" or _is_symbol_like_token(token):
            node = node.next
            continue
        if replace_num and _is_numeric_token(token):
            token = "<NUM>"
        tokens.append(token)
        node = node.next

    return tokens


def _split_segments(text: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return [text]
    return [s.strip() for s in text.split(delimiter) if s.strip()]


def tokenize_japanese_document_tokens(
    text: str,
    *,
    delimiter: str | None = " / ",
    replace_num: bool = True,
    stopwords: set[str] | None = None,
    dicdir: str | None = None,
    require_unidic: bool = True,
) -> list[str]:
    tokens: list[str] = []
    for segment in _split_segments(text, delimiter):
        tokens.extend(
            tokenize_japanese_text(
                segment,
                replace_num=replace_num,
                stopwords=stopwords,
                dicdir=dicdir,
                require_unidic=require_unidic,
            )
        )
    return tokens


def tokenize_japanese_sentence_strings(
    text: str,
    *,
    delimiter: str | None = " / ",
    replace_num: bool = True,
    stopwords: set[str] | None = None,
    dicdir: str | None = None,
    require_unidic: bool = True,
) -> list[str]:
    sentences: list[str] = []
    for segment in _split_segments(text, delimiter):
        tokens = tokenize_japanese_text(
            segment,
            replace_num=replace_num,
            stopwords=stopwords,
            dicdir=dicdir,
            require_unidic=require_unidic,
        )
        if tokens:
            sentences.append(" ".join(tokens))
    return sentences


def tokenize_japanese_documents(
    docs: Sequence[str],
    *,
    delimiter: str | None = " / ",
    replace_num: bool = True,
    stopwords: set[str] | None = None,
    dicdir: str | None = None,
    require_unidic: bool = True,
) -> list[list[str]]:
    return [
        tokenize_japanese_document_tokens(
            doc,
            delimiter=delimiter,
            replace_num=replace_num,
            stopwords=stopwords,
            dicdir=dicdir,
            require_unidic=require_unidic,
        )
        for doc in docs
    ]
