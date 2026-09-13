"""Systematic all-topic-pair analysis in the shared sentence-embedding space.

For every trained run of the sentence-level models (vMF Sentence LDA, SentLDA,
Sentence Gaussian LDA) the same procedure re-estimates the sentence-topic
posteriors with the collapsed fold-in of the representative-word protocol,
re-encodes the training sentences once per category, and records per-topic
concentration, centroid geometry, assignment confusion and fine-label
divergence for every pair of topics, plus the cross-model topic overlap.

Raw per-run values leave through ``*.scores.json`` sidecars
(:mod:`.summary`); all aggregation happens downstream.
"""
