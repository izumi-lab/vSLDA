from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .sentence_topic_scoring import normalize_rows, vmf_log_density_all_topics


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_doc_topics(
    doc_topics: np.ndarray,
    out_path: Path,
    *,
    seed: int | None = None,
    title: str = "Doc-topic TSNE (color = argmax topic)",
) -> bool:
    if doc_topics.shape[0] == 0:
        return False
    if np.isnan(doc_topics).any() or np.allclose(doc_topics.std(axis=0), 0.0):
        return False

    n_samples = doc_topics.shape[0]
    perplexity = min(30, max(2, n_samples - 1))
    coords = TSNE(
        n_components=2, random_state=seed, perplexity=perplexity
    ).fit_transform(doc_topics)
    labels = np.argmax(doc_topics, axis=1)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=8)
    plt.title(title)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.colorbar(scatter, label="Topic")
    ensure_directory(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def plot_average_ll(avg_ll: Sequence[float], out_path: Path) -> bool:
    if not avg_ll:
        return False

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(avg_ll)), avg_ll, marker="o")
    plt.title("Average Log-Likelihood")
    plt.xlabel("Iteration")
    plt.ylabel("Avg LL")
    ensure_directory(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _create_sphere_mesh(
    num_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0, np.pi, num_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi, theta = np.meshgrid(phi, theta)
    xs = np.sin(phi) * np.cos(theta)
    ys = np.sin(phi) * np.sin(theta)
    zs = np.cos(phi)
    return xs, ys, zs


def _set_equal_aspect_3d(ax, points: np.ndarray) -> None:
    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_embeddings_on_sphere_3d(
    *,
    embeddings: np.ndarray,
    topic_means: np.ndarray,
    kappa_per_topic: np.ndarray,
    mixture_weights: np.ndarray,
    component_means: np.ndarray,
    out_path: Path,
    max_points: int = 2000,
    seed: int | None = None,
) -> dict[str, int]:
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings to plot.")
    if embeddings.shape[1] != topic_means.shape[1]:
        raise ValueError(
            f"Dim mismatch: embeddings {embeddings.shape[1]} vs topic_means {topic_means.shape[1]}"
        )

    rng = np.random.default_rng(seed)
    if embeddings.shape[0] > max_points:
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        sampled_embeddings = embeddings[idx]
    else:
        sampled_embeddings = embeddings

    log_scores = vmf_log_density_all_topics(
        embeddings=sampled_embeddings,
        topic_means=topic_means,
        kappa_per_topic=kappa_per_topic,
        mixture_weights=mixture_weights,
        component_means=component_means,
    )
    labels = np.argmax(log_scores, axis=1)

    pca = PCA(n_components=3)
    emb_3d = normalize_rows(pca.fit_transform(sampled_embeddings))
    topic_3d = normalize_rows(pca.transform(topic_means))
    xs, ys, zs = _create_sphere_mesh(num_points=50)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.3)
    scatter = ax.scatter(
        emb_3d[:, 0],
        emb_3d[:, 1],
        emb_3d[:, 2],
        s=5,
        alpha=0.4,
        c=labels,
        cmap="tab20",
    )
    ax.scatter(
        topic_3d[:, 0],
        topic_3d[:, 1],
        topic_3d[:, 2],
        s=60,
        marker="^",
    )
    ax.set_title("Sentence embeddings & topic means on 3D sphere (PCA-projected)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Topic")
    _set_equal_aspect_3d(ax, np.vstack([emb_3d, topic_3d]))
    ensure_directory(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return {
        "num_embeddings_plotted": int(sampled_embeddings.shape[0]),
        "num_topics": int(topic_means.shape[0]),
    }
