from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Tuple

from gensim.models import KeyedVectors

from src.core.artifacts import copy_binary_stream_to_path, extract_bz2_file
from src.core.paths import WIKIENTVEC_ROOT

DEFAULT_CACHE_DIR = WIKIENTVEC_ROOT
DEFAULT_TAG = "20190520"
DEFAULT_ASSET = "jawiki.word_vectors.200d.txt.bz2"

LOGGER = logging.getLogger(__name__)


def is_wikientvec_spec(spec: str | None) -> bool:
    if not spec:
        return False
    return spec.strip().lower().startswith("wikientvec")


def _parse_wikientvec_spec(spec: str) -> Tuple[str, str]:
    """
    Supported forms:
      - wikientvec
      - wikientvec:<tag>
      - wikientvec:<tag>:<asset>
      - wikientvec://<tag>/<asset>
    """
    s = spec.strip()
    if s.lower() == "wikientvec":
        return DEFAULT_TAG, DEFAULT_ASSET

    if s.lower().startswith("wikientvec://"):
        payload = s[len("wikientvec://") :]
        parts = payload.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
        raise ValueError(f"Invalid WikiEntVec spec: {spec}")

    if s.lower().startswith("wikientvec:"):
        payload = s[len("wikientvec:") :]
        parts = payload.split(":")
        if len(parts) == 1 and parts[0]:
            return parts[0], DEFAULT_ASSET
        if len(parts) >= 2 and parts[0] and parts[1]:
            return parts[0], ":".join(parts[1:])
        raise ValueError(f"Invalid WikiEntVec spec: {spec}")

    raise ValueError(f"Invalid WikiEntVec spec: {spec}")


def _download(url: str, dst: Path) -> None:
    with urllib.request.urlopen(url) as src:
        copy_binary_stream_to_path(src, dst)


def _extract_bz2(src: Path, dst: Path) -> None:
    extract_bz2_file(src, dst)


def ensure_wikientvec_file(
    spec: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    tag, asset = _parse_wikientvec_spec(spec)
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    compressed_path = out_dir / asset
    vector_path = out_dir / asset.removesuffix(".bz2")

    if vector_path.exists():
        return vector_path

    if not compressed_path.exists():
        url = (
            "https://github.com/singletongue/WikiEntVec/releases/download/"
            f"{tag}/{asset}"
        )
        LOGGER.info("WikiEntVec: downloading %s", url)
        _download(url, compressed_path)

    if asset.endswith(".bz2"):
        LOGGER.info("WikiEntVec: extracting %s -> %s", compressed_path, vector_path)
        _extract_bz2(compressed_path, vector_path)
    else:
        vector_path = compressed_path

    if not vector_path.exists():
        raise FileNotFoundError(
            f"WikiEntVec vector file was not prepared: {vector_path}"
        )
    return vector_path


def load_wikientvec(
    spec: str,
    *,
    cache_dir: str | Path | None = None,
) -> KeyedVectors:
    vector_path = ensure_wikientvec_file(spec, cache_dir=cache_dir)
    LOGGER.info("WikiEntVec: loading vectors from %s", vector_path)
    return KeyedVectors.load_word2vec_format(vector_path.as_posix(), binary=False)
