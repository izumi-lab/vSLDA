from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency fallback
    njit = None
    NUMBA_AVAILABLE = False


def resolve_sentlda_backend(preferred: str) -> str:
    backend = str(preferred).strip().lower()
    if backend not in {"auto", "python", "numba"}:
        raise ValueError("sentLDA backend must be one of 'auto', 'python', or 'numba'.")
    if backend == "auto":
        return "numba" if NUMBA_AVAILABLE else "python"
    if backend == "numba" and not NUMBA_AVAILABLE:
        raise RuntimeError(
            "sentLDA backend='numba' requested, but numba is not available."
        )
    return backend


def _compute_sentence_topic_distribution_python(
    doc_topic_counts: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    unique_start: int,
    unique_end: int,
    sentence_length: int,
    alpha: float,
    beta: float,
    vocab_size: int,
    out_probs: np.ndarray,
) -> None:
    num_topics = int(doc_topic_counts.shape[0])
    log_prior = np.empty(num_topics, dtype=np.float64)
    log_likelihood = np.empty(num_topics, dtype=np.float64)
    _compute_sentence_topic_log_factors_python(
        doc_topic_counts=doc_topic_counts,
        topic_word_counts=topic_word_counts,
        topic_total_words=topic_total_words,
        sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
        sentence_word_counts_flat=sentence_word_counts_flat,
        unique_start=unique_start,
        unique_end=unique_end,
        sentence_length=sentence_length,
        alpha=alpha,
        beta=beta,
        vocab_size=vocab_size,
        out_log_prior=log_prior,
        out_log_likelihood=log_likelihood,
    )
    _softmax_log_scores_python(log_prior + log_likelihood, out_probs)


def _compute_sentence_topic_log_factors_python(
    doc_topic_counts: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    unique_start: int,
    unique_end: int,
    sentence_length: int,
    alpha: float,
    beta: float,
    vocab_size: int,
    out_log_prior: np.ndarray,
    out_log_likelihood: np.ndarray,
) -> None:
    num_topics = int(doc_topic_counts.shape[0])
    beta_vocab = float(beta) * float(vocab_size)

    for topic in range(num_topics):
        prior_mass = float(doc_topic_counts[topic]) + float(alpha)
        if prior_mass <= 0.0:
            out_log_prior[topic] = -1.0e300
        else:
            out_log_prior[topic] = math.log(prior_mass)

        log_value = 0.0
        running_total = float(topic_total_words[topic]) + beta_vocab
        if running_total <= 0.0:
            log_value = -1.0e300
        else:
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                word_mass = float(topic_word_counts[topic, word_id]) + float(beta)
                if word_mass <= 0.0:
                    log_value = -1.0e300
                    break
                for offset in range(count):
                    log_value += math.log(word_mass + float(offset))
            if log_value > -1.0e299:
                for offset in range(sentence_length):
                    log_value -= math.log(running_total + float(offset))

        out_log_likelihood[topic] = log_value


def _softmax_log_scores_python(log_scores: np.ndarray, out_probs: np.ndarray) -> None:
    num_topics = int(log_scores.shape[0])
    max_log = -1.0e300
    for topic in range(num_topics):
        log_value = float(log_scores[topic])
        if log_value > max_log:
            max_log = log_value
    if (not math.isfinite(max_log)) or max_log <= -1.0e299:
        out_probs.fill(1.0 / float(num_topics))
        return

    score_sum = 0.0
    for topic in range(num_topics):
        value = math.exp(log_scores[topic] - max_log)
        out_probs[topic] = value
        score_sum += value

    if (not math.isfinite(score_sum)) or score_sum <= 0.0:
        out_probs.fill(1.0 / float(num_topics))
        return

    inv_sum = 1.0 / score_sum
    for topic in range(num_topics):
        out_probs[topic] *= inv_sum


def _sample_topic_from_probs_python(probs: np.ndarray, uniform: float) -> int:
    if probs.size == 0:
        return 0
    threshold = min(max(float(uniform), 0.0), 1.0) * float(probs.sum())
    cumulative = 0.0
    for topic in range(int(probs.shape[0])):
        cumulative += float(probs[topic])
        if threshold <= cumulative:
            return topic
    return int(probs.shape[0]) - 1


def _run_sentlda_train_iteration_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    uniforms: np.ndarray,
) -> None:
    num_topics = int(doc_topic_counts.shape[1])
    probs = np.empty(num_topics, dtype=np.float64)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            old_topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, old_topic] -= 1
            topic_total_words[old_topic] -= sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[old_topic, word_id] -= count

            _compute_sentence_topic_distribution_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_probs=probs,
            )
            new_topic = _sample_topic_from_probs_python(
                probs=probs,
                uniform=float(uniforms[sentence_index]),
            )
            assignments[sentence_index] = new_topic
            doc_topic_counts[doc_index, new_topic] += 1
            topic_total_words[new_topic] += sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[new_topic, word_id] += count


def _run_sentlda_infer_iteration_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    uniforms: np.ndarray,
) -> None:
    num_topics = int(doc_topic_counts.shape[1])
    probs = np.empty(num_topics, dtype=np.float64)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            old_topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, old_topic] -= 1
            _compute_sentence_topic_distribution_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_probs=probs,
            )
            new_topic = _sample_topic_from_probs_python(
                probs=probs,
                uniform=float(uniforms[sentence_index]),
            )
            assignments[sentence_index] = new_topic
            doc_topic_counts[doc_index, new_topic] += 1


def _build_sentence_topic_soft_train_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
) -> np.ndarray:
    num_sentences = int(sentence_lengths.shape[0])
    num_topics = int(doc_topic_counts.shape[1])
    probs = np.empty(num_topics, dtype=np.float64)
    output = np.empty((num_sentences, num_topics), dtype=np.float32)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, topic] -= 1
            topic_total_words[topic] -= sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[topic, word_id] -= count

            _compute_sentence_topic_distribution_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_probs=probs,
            )
            for topic_index in range(num_topics):
                output[sentence_index, topic_index] = np.float32(probs[topic_index])

            doc_topic_counts[doc_index, topic] += 1
            topic_total_words[topic] += sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[topic, word_id] += count

    return output


def _build_sentence_topic_soft_infer_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
) -> np.ndarray:
    num_sentences = int(sentence_lengths.shape[0])
    num_topics = int(doc_topic_counts.shape[1])
    probs = np.empty(num_topics, dtype=np.float64)
    output = np.empty((num_sentences, num_topics), dtype=np.float32)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, topic] -= 1
            _compute_sentence_topic_distribution_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_probs=probs,
            )
            for topic_index in range(num_topics):
                output[sentence_index, topic_index] = np.float32(probs[topic_index])
            doc_topic_counts[doc_index, topic] += 1

    return output


def _build_sentence_topic_log_factors_train_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_sentences = int(sentence_lengths.shape[0])
    num_topics = int(doc_topic_counts.shape[1])
    log_prior = np.empty((num_sentences, num_topics), dtype=np.float32)
    log_likelihood = np.empty((num_sentences, num_topics), dtype=np.float32)
    row_prior = np.empty(num_topics, dtype=np.float64)
    row_likelihood = np.empty(num_topics, dtype=np.float64)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, topic] -= 1
            topic_total_words[topic] -= sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[topic, word_id] -= count

            _compute_sentence_topic_log_factors_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_log_prior=row_prior,
                out_log_likelihood=row_likelihood,
            )
            for topic_index in range(num_topics):
                log_prior[sentence_index, topic_index] = np.float32(
                    row_prior[topic_index]
                )
                log_likelihood[sentence_index, topic_index] = np.float32(
                    row_likelihood[topic_index]
                )

            doc_topic_counts[doc_index, topic] += 1
            topic_total_words[topic] += sentence_length
            for flat_index in range(unique_start, unique_end):
                word_id = int(sentence_unique_word_ids_flat[flat_index])
                count = int(sentence_word_counts_flat[flat_index])
                topic_word_counts[topic, word_id] += count

    return log_prior, log_likelihood


def _build_sentence_topic_log_factors_infer_python(
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_sentences = int(sentence_lengths.shape[0])
    num_topics = int(doc_topic_counts.shape[1])
    log_prior = np.empty((num_sentences, num_topics), dtype=np.float32)
    log_likelihood = np.empty((num_sentences, num_topics), dtype=np.float32)
    row_prior = np.empty(num_topics, dtype=np.float64)
    row_likelihood = np.empty(num_topics, dtype=np.float64)

    for doc_index in range(int(doc_offsets.shape[0]) - 1):
        sentence_start = int(doc_offsets[doc_index])
        sentence_end = int(doc_offsets[doc_index + 1])
        for sentence_index in range(sentence_start, sentence_end):
            topic = int(assignments[sentence_index])
            sentence_length = int(sentence_lengths[sentence_index])
            unique_start = int(sentence_unique_offsets[sentence_index])
            unique_end = int(sentence_unique_offsets[sentence_index + 1])

            doc_topic_counts[doc_index, topic] -= 1
            _compute_sentence_topic_log_factors_python(
                doc_topic_counts=doc_topic_counts[doc_index],
                topic_word_counts=topic_word_counts,
                topic_total_words=topic_total_words,
                sentence_unique_word_ids_flat=sentence_unique_word_ids_flat,
                sentence_word_counts_flat=sentence_word_counts_flat,
                unique_start=unique_start,
                unique_end=unique_end,
                sentence_length=sentence_length,
                alpha=alpha,
                beta=beta,
                vocab_size=vocab_size,
                out_log_prior=row_prior,
                out_log_likelihood=row_likelihood,
            )
            for topic_index in range(num_topics):
                log_prior[sentence_index, topic_index] = np.float32(
                    row_prior[topic_index]
                )
                log_likelihood[sentence_index, topic_index] = np.float32(
                    row_likelihood[topic_index]
                )
            doc_topic_counts[doc_index, topic] += 1

    return log_prior, log_likelihood


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _compute_sentence_topic_distribution_numba(
        doc_topic_counts: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_total_words: np.ndarray,
        sentence_unique_word_ids_flat: np.ndarray,
        sentence_word_counts_flat: np.ndarray,
        unique_start: int,
        unique_end: int,
        sentence_length: int,
        alpha: float,
        beta: float,
        vocab_size: int,
        out_probs: np.ndarray,
    ) -> None:
        num_topics = doc_topic_counts.shape[0]
        log_scores = np.empty(num_topics, dtype=np.float64)
        max_log = -1.0e300
        beta_vocab = float(beta) * float(vocab_size)

        for topic in range(num_topics):
            prior_mass = float(doc_topic_counts[topic]) + float(alpha)
            if prior_mass <= 0.0:
                log_value = -1.0e300
            else:
                log_value = math.log(prior_mass)
                running_total = float(topic_total_words[topic]) + beta_vocab
                if running_total <= 0.0:
                    log_value = -1.0e300
                else:
                    for flat_index in range(unique_start, unique_end):
                        word_id = int(sentence_unique_word_ids_flat[flat_index])
                        count = int(sentence_word_counts_flat[flat_index])
                        word_mass = float(topic_word_counts[topic, word_id]) + float(
                            beta
                        )
                        if word_mass <= 0.0:
                            log_value = -1.0e300
                            break
                        for offset in range(count):
                            log_value += math.log(word_mass + float(offset))
                    if log_value > -1.0e299:
                        for offset in range(sentence_length):
                            log_value -= math.log(running_total + float(offset))

            log_scores[topic] = log_value
            if log_value > max_log:
                max_log = log_value

        if (not math.isfinite(max_log)) or max_log <= -1.0e299:
            for topic in range(num_topics):
                out_probs[topic] = 1.0 / float(num_topics)
            return

        score_sum = 0.0
        for topic in range(num_topics):
            value = math.exp(log_scores[topic] - max_log)
            out_probs[topic] = value
            score_sum += value

        if (not math.isfinite(score_sum)) or score_sum <= 0.0:
            for topic in range(num_topics):
                out_probs[topic] = 1.0 / float(num_topics)
            return

        inv_sum = 1.0 / score_sum
        for topic in range(num_topics):
            out_probs[topic] *= inv_sum

    @njit(cache=True)
    def _sample_topic_from_probs_numba(probs: np.ndarray, uniform: float) -> int:
        if probs.shape[0] == 0:
            return 0
        clipped = min(max(float(uniform), 0.0), 1.0)
        threshold = clipped * float(np.sum(probs))
        cumulative = 0.0
        for topic in range(probs.shape[0]):
            cumulative += float(probs[topic])
            if threshold <= cumulative:
                return topic
        return probs.shape[0] - 1

    @njit(cache=True)
    def _run_sentlda_train_iteration_numba(
        doc_offsets: np.ndarray,
        sentence_lengths: np.ndarray,
        sentence_unique_offsets: np.ndarray,
        sentence_unique_word_ids_flat: np.ndarray,
        sentence_word_counts_flat: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_total_words: np.ndarray,
        doc_topic_counts: np.ndarray,
        assignments: np.ndarray,
        alpha: float,
        beta: float,
        vocab_size: int,
        uniforms: np.ndarray,
    ) -> None:
        num_topics = doc_topic_counts.shape[1]
        probs = np.empty(num_topics, dtype=np.float64)

        for doc_index in range(doc_offsets.shape[0] - 1):
            sentence_start = int(doc_offsets[doc_index])
            sentence_end = int(doc_offsets[doc_index + 1])
            for sentence_index in range(sentence_start, sentence_end):
                old_topic = int(assignments[sentence_index])
                sentence_length = int(sentence_lengths[sentence_index])
                unique_start = int(sentence_unique_offsets[sentence_index])
                unique_end = int(sentence_unique_offsets[sentence_index + 1])

                doc_topic_counts[doc_index, old_topic] -= 1
                topic_total_words[old_topic] -= sentence_length
                for flat_index in range(unique_start, unique_end):
                    word_id = int(sentence_unique_word_ids_flat[flat_index])
                    count = int(sentence_word_counts_flat[flat_index])
                    topic_word_counts[old_topic, word_id] -= count

                _compute_sentence_topic_distribution_numba(
                    doc_topic_counts[doc_index],
                    topic_word_counts,
                    topic_total_words,
                    sentence_unique_word_ids_flat,
                    sentence_word_counts_flat,
                    unique_start,
                    unique_end,
                    sentence_length,
                    alpha,
                    beta,
                    vocab_size,
                    probs,
                )
                new_topic = _sample_topic_from_probs_numba(
                    probs,
                    float(uniforms[sentence_index]),
                )
                assignments[sentence_index] = new_topic
                doc_topic_counts[doc_index, new_topic] += 1
                topic_total_words[new_topic] += sentence_length
                for flat_index in range(unique_start, unique_end):
                    word_id = int(sentence_unique_word_ids_flat[flat_index])
                    count = int(sentence_word_counts_flat[flat_index])
                    topic_word_counts[new_topic, word_id] += count

    @njit(cache=True)
    def _run_sentlda_infer_iteration_numba(
        doc_offsets: np.ndarray,
        sentence_lengths: np.ndarray,
        sentence_unique_offsets: np.ndarray,
        sentence_unique_word_ids_flat: np.ndarray,
        sentence_word_counts_flat: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_total_words: np.ndarray,
        doc_topic_counts: np.ndarray,
        assignments: np.ndarray,
        alpha: float,
        beta: float,
        vocab_size: int,
        uniforms: np.ndarray,
    ) -> None:
        num_topics = doc_topic_counts.shape[1]
        probs = np.empty(num_topics, dtype=np.float64)

        for doc_index in range(doc_offsets.shape[0] - 1):
            sentence_start = int(doc_offsets[doc_index])
            sentence_end = int(doc_offsets[doc_index + 1])
            for sentence_index in range(sentence_start, sentence_end):
                old_topic = int(assignments[sentence_index])
                sentence_length = int(sentence_lengths[sentence_index])
                unique_start = int(sentence_unique_offsets[sentence_index])
                unique_end = int(sentence_unique_offsets[sentence_index + 1])

                doc_topic_counts[doc_index, old_topic] -= 1
                _compute_sentence_topic_distribution_numba(
                    doc_topic_counts[doc_index],
                    topic_word_counts,
                    topic_total_words,
                    sentence_unique_word_ids_flat,
                    sentence_word_counts_flat,
                    unique_start,
                    unique_end,
                    sentence_length,
                    alpha,
                    beta,
                    vocab_size,
                    probs,
                )
                new_topic = _sample_topic_from_probs_numba(
                    probs,
                    float(uniforms[sentence_index]),
                )
                assignments[sentence_index] = new_topic
                doc_topic_counts[doc_index, new_topic] += 1

    @njit(cache=True)
    def _build_sentence_topic_soft_train_numba(
        doc_offsets: np.ndarray,
        sentence_lengths: np.ndarray,
        sentence_unique_offsets: np.ndarray,
        sentence_unique_word_ids_flat: np.ndarray,
        sentence_word_counts_flat: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_total_words: np.ndarray,
        doc_topic_counts: np.ndarray,
        assignments: np.ndarray,
        alpha: float,
        beta: float,
        vocab_size: int,
    ) -> np.ndarray:
        num_sentences = sentence_lengths.shape[0]
        num_topics = doc_topic_counts.shape[1]
        probs = np.empty(num_topics, dtype=np.float64)
        output = np.empty((num_sentences, num_topics), dtype=np.float32)

        for doc_index in range(doc_offsets.shape[0] - 1):
            sentence_start = int(doc_offsets[doc_index])
            sentence_end = int(doc_offsets[doc_index + 1])
            for sentence_index in range(sentence_start, sentence_end):
                topic = int(assignments[sentence_index])
                sentence_length = int(sentence_lengths[sentence_index])
                unique_start = int(sentence_unique_offsets[sentence_index])
                unique_end = int(sentence_unique_offsets[sentence_index + 1])

                doc_topic_counts[doc_index, topic] -= 1
                topic_total_words[topic] -= sentence_length
                for flat_index in range(unique_start, unique_end):
                    word_id = int(sentence_unique_word_ids_flat[flat_index])
                    count = int(sentence_word_counts_flat[flat_index])
                    topic_word_counts[topic, word_id] -= count

                _compute_sentence_topic_distribution_numba(
                    doc_topic_counts[doc_index],
                    topic_word_counts,
                    topic_total_words,
                    sentence_unique_word_ids_flat,
                    sentence_word_counts_flat,
                    unique_start,
                    unique_end,
                    sentence_length,
                    alpha,
                    beta,
                    vocab_size,
                    probs,
                )
                for topic_index in range(num_topics):
                    output[sentence_index, topic_index] = np.float32(probs[topic_index])

                doc_topic_counts[doc_index, topic] += 1
                topic_total_words[topic] += sentence_length
                for flat_index in range(unique_start, unique_end):
                    word_id = int(sentence_unique_word_ids_flat[flat_index])
                    count = int(sentence_word_counts_flat[flat_index])
                    topic_word_counts[topic, word_id] += count

        return output

    @njit(cache=True)
    def _build_sentence_topic_soft_infer_numba(
        doc_offsets: np.ndarray,
        sentence_lengths: np.ndarray,
        sentence_unique_offsets: np.ndarray,
        sentence_unique_word_ids_flat: np.ndarray,
        sentence_word_counts_flat: np.ndarray,
        topic_word_counts: np.ndarray,
        topic_total_words: np.ndarray,
        doc_topic_counts: np.ndarray,
        assignments: np.ndarray,
        alpha: float,
        beta: float,
        vocab_size: int,
    ) -> np.ndarray:
        num_sentences = sentence_lengths.shape[0]
        num_topics = doc_topic_counts.shape[1]
        probs = np.empty(num_topics, dtype=np.float64)
        output = np.empty((num_sentences, num_topics), dtype=np.float32)

        for doc_index in range(doc_offsets.shape[0] - 1):
            sentence_start = int(doc_offsets[doc_index])
            sentence_end = int(doc_offsets[doc_index + 1])
            for sentence_index in range(sentence_start, sentence_end):
                topic = int(assignments[sentence_index])
                sentence_length = int(sentence_lengths[sentence_index])
                unique_start = int(sentence_unique_offsets[sentence_index])
                unique_end = int(sentence_unique_offsets[sentence_index + 1])

                doc_topic_counts[doc_index, topic] -= 1
                _compute_sentence_topic_distribution_numba(
                    doc_topic_counts[doc_index],
                    topic_word_counts,
                    topic_total_words,
                    sentence_unique_word_ids_flat,
                    sentence_word_counts_flat,
                    unique_start,
                    unique_end,
                    sentence_length,
                    alpha,
                    beta,
                    vocab_size,
                    probs,
                )
                for topic_index in range(num_topics):
                    output[sentence_index, topic_index] = np.float32(probs[topic_index])
                doc_topic_counts[doc_index, topic] += 1

        return output

else:  # pragma: no cover - optional dependency fallback
    _compute_sentence_topic_distribution_numba = None
    _sample_topic_from_probs_numba = None
    _run_sentlda_train_iteration_numba = None
    _run_sentlda_infer_iteration_numba = None
    _build_sentence_topic_soft_train_numba = None
    _build_sentence_topic_soft_infer_numba = None


def run_sentlda_train_iteration(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    uniforms: np.ndarray,
    backend: str,
) -> None:
    if backend == "numba":
        _run_sentlda_train_iteration_numba(
            np.asarray(doc_offsets, dtype=np.int32),
            np.asarray(sentence_lengths, dtype=np.int32),
            np.asarray(sentence_unique_offsets, dtype=np.int32),
            np.asarray(sentence_unique_word_ids_flat, dtype=np.int32),
            np.asarray(sentence_word_counts_flat, dtype=np.int32),
            np.asarray(topic_word_counts, dtype=np.int64),
            np.asarray(topic_total_words, dtype=np.int64),
            np.asarray(doc_topic_counts, dtype=np.int64),
            np.asarray(assignments, dtype=np.int32),
            float(alpha),
            float(beta),
            int(vocab_size),
            np.asarray(uniforms, dtype=np.float64),
        )
        return

    _run_sentlda_train_iteration_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
        uniforms=np.asarray(uniforms, dtype=np.float64),
    )


def run_sentlda_infer_iteration(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    uniforms: np.ndarray,
    backend: str,
) -> None:
    if backend == "numba":
        _run_sentlda_infer_iteration_numba(
            np.asarray(doc_offsets, dtype=np.int32),
            np.asarray(sentence_lengths, dtype=np.int32),
            np.asarray(sentence_unique_offsets, dtype=np.int32),
            np.asarray(sentence_unique_word_ids_flat, dtype=np.int32),
            np.asarray(sentence_word_counts_flat, dtype=np.int32),
            np.asarray(topic_word_counts, dtype=np.int64),
            np.asarray(topic_total_words, dtype=np.int64),
            np.asarray(doc_topic_counts, dtype=np.int64),
            np.asarray(assignments, dtype=np.int32),
            float(alpha),
            float(beta),
            int(vocab_size),
            np.asarray(uniforms, dtype=np.float64),
        )
        return

    _run_sentlda_infer_iteration_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
        uniforms=np.asarray(uniforms, dtype=np.float64),
    )


def build_sentence_topic_soft_train(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    backend: str,
) -> np.ndarray:
    if backend == "numba":
        return _build_sentence_topic_soft_train_numba(
            np.asarray(doc_offsets, dtype=np.int32),
            np.asarray(sentence_lengths, dtype=np.int32),
            np.asarray(sentence_unique_offsets, dtype=np.int32),
            np.asarray(sentence_unique_word_ids_flat, dtype=np.int32),
            np.asarray(sentence_word_counts_flat, dtype=np.int32),
            np.asarray(topic_word_counts, dtype=np.int64),
            np.asarray(topic_total_words, dtype=np.int64),
            np.asarray(doc_topic_counts, dtype=np.int64),
            np.asarray(assignments, dtype=np.int32),
            float(alpha),
            float(beta),
            int(vocab_size),
        )
    return _build_sentence_topic_soft_train_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
    )


def build_sentence_topic_soft_infer(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    backend: str,
) -> np.ndarray:
    if backend == "numba":
        return _build_sentence_topic_soft_infer_numba(
            np.asarray(doc_offsets, dtype=np.int32),
            np.asarray(sentence_lengths, dtype=np.int32),
            np.asarray(sentence_unique_offsets, dtype=np.int32),
            np.asarray(sentence_unique_word_ids_flat, dtype=np.int32),
            np.asarray(sentence_word_counts_flat, dtype=np.int32),
            np.asarray(topic_word_counts, dtype=np.int64),
            np.asarray(topic_total_words, dtype=np.int64),
            np.asarray(doc_topic_counts, dtype=np.int64),
            np.asarray(assignments, dtype=np.int32),
            float(alpha),
            float(beta),
            int(vocab_size),
        )
    return _build_sentence_topic_soft_infer_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
    )


def build_sentence_topic_log_factors_train(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    _ = backend
    return _build_sentence_topic_log_factors_train_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
    )


def build_sentence_topic_log_factors_infer(
    *,
    doc_offsets: np.ndarray,
    sentence_lengths: np.ndarray,
    sentence_unique_offsets: np.ndarray,
    sentence_unique_word_ids_flat: np.ndarray,
    sentence_word_counts_flat: np.ndarray,
    topic_word_counts: np.ndarray,
    topic_total_words: np.ndarray,
    doc_topic_counts: np.ndarray,
    assignments: np.ndarray,
    alpha: float,
    beta: float,
    vocab_size: int,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    _ = backend
    return _build_sentence_topic_log_factors_infer_python(
        doc_offsets=np.asarray(doc_offsets, dtype=np.int32),
        sentence_lengths=np.asarray(sentence_lengths, dtype=np.int32),
        sentence_unique_offsets=np.asarray(sentence_unique_offsets, dtype=np.int32),
        sentence_unique_word_ids_flat=np.asarray(
            sentence_unique_word_ids_flat,
            dtype=np.int32,
        ),
        sentence_word_counts_flat=np.asarray(sentence_word_counts_flat, dtype=np.int32),
        topic_word_counts=np.asarray(topic_word_counts, dtype=np.int64),
        topic_total_words=np.asarray(topic_total_words, dtype=np.int64),
        doc_topic_counts=np.asarray(doc_topic_counts, dtype=np.int64),
        assignments=np.asarray(assignments, dtype=np.int32),
        alpha=float(alpha),
        beta=float(beta),
        vocab_size=int(vocab_size),
    )


SENTLDA_NUMERICS_BACKEND = "numba" if NUMBA_AVAILABLE else "python"


__all__ = [
    "NUMBA_AVAILABLE",
    "SENTLDA_NUMERICS_BACKEND",
    "build_sentence_topic_log_factors_infer",
    "build_sentence_topic_log_factors_train",
    "build_sentence_topic_soft_infer",
    "build_sentence_topic_soft_train",
    "resolve_sentlda_backend",
    "run_sentlda_infer_iteration",
    "run_sentlda_train_iteration",
]
