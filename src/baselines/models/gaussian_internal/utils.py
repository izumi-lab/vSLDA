from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import progressbar as pb
from numpy.random import default_rng


def _load_choldate() -> tuple[
    Callable[[np.ndarray, np.ndarray], None],
    Callable[[np.ndarray, np.ndarray], None],
]:
    try:
        from choldate import choldowndate, cholupdate
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gaussian trainers require the optional dependency `choldate` for "
            "rank-1 Cholesky updates. Install it before running Gaussian-family training."
        ) from exc
    return choldowndate, cholupdate


def get_logger(name):
    level = logging.INFO
    log = logging.getLogger(name)
    log.setLevel(level)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    try:
        import coloredlogs
    except ImportError:
        if not log.handlers:
            sh = logging.StreamHandler()
            sh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt)
            sh.setFormatter(formatter)
            log.addHandler(sh)
    else:
        coloredlogs.install(level=level, log=log, fmt=fmt)

    return log


def get_progress_bar(maxval, title=None, counter=False, show_progress=True):
    if not show_progress:
        return lambda x: x

    widgets = []
    if title is not None:
        widgets.append("%s: " % title)
    if maxval is not pb.UnknownLength:
        widgets.extend([pb.Percentage(), " ", pb.Bar(marker=pb.RotatingMarker())])
    if counter:
        widgets.extend([" (", pb.Counter(), ")"])
    if maxval is not pb.UnknownLength:
        widgets.extend([" ", pb.ETA()])
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxval)
    return pbar


def chol_rank1_update(L, x):
    _choldowndate, cholupdate = _load_choldate()
    cholupdate(L.T, x.copy())


def chol_rank1_downdate(L, x):
    choldowndate, _cholupdate = _load_choldate()
    choldowndate(L.T, x.copy())


def sum_logprobs(logprobs):
    max_prob = logprobs.max()
    return np.log(np.sum(np.exp(logprobs - max_prob))) + max_prob


class BatchedRands:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.rng = default_rng()
        self._it = iter(self)

    def __iter__(self):
        while True:
            batch = self.rng.random(self.batch_size)
            for v in batch:
                yield v

    def random(self):
        return next(self._it)

    def integer(self, high):
        return int(high * self.random())

    def choice(self, p):
        rand = self.random()
        return np.argmax(np.cumsum(p) > rand)

    def choice_cum(self, p):
        rand = self.random()
        return np.argmax(p > rand)


class BatchedRandInts:
    def __init__(self, high, batch_size=1000):
        self.high = high
        self.batch_size = batch_size
        self.rng = default_rng()
        self._new_batch()

    def _new_batch(self):
        self._current_id = 0
        self._batch = self.rng.integers(self.high, size=self.batch_size)

    def integers(self, size):
        self._current_id += size
        if self._current_id > self.batch_size:
            self._new_batch()
            self._current_id += size

        return self._batch[self._current_id - size : self._current_id]
