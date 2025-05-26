import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def rotate(seq: np.ndarray, angle_rad: float, origin=(0, 0)):
    """
    seq: np.ndarray of shape [sequence_length, 2] for xy sequence
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    translated = seq - origin
    rotated = translated @ rotation_matrix.T
    return rotated + origin


def resample_sequence(seq: np.ndarray, num_points=32):
    """
    seq: np.ndarray of shape [sequence_length, 2] for xy sequence
    """
    deltas = np.diff(seq, axis=0)
    dist = np.sqrt((deltas**2).sum(axis=1))
    cumulative_dist = np.insert(np.cumsum(dist), 0, 0)

    interp_func = interp1d(
        cumulative_dist, seq, kind="linear", axis=0, assume_sorted=True
    )
    new_dist = np.linspace(0, cumulative_dist[-1], num_points)
    return interp_func(new_dist)


def center_sequences(seq: np.ndarray) -> np.ndarray:
    """
    seq: np.ndarray of shape [batch, sequence_length, 2] for xy sequences
    """
    center = (np.max(seq, axis=1) + np.min(seq, axis=1)) / 2
    return seq - center[:, None, :]


def vectorize(seq: np.ndarray, orientation_sensitive: bool) -> np.ndarray:
    """
    seq: np.ndarray of shape [batch, sequence_length, 2] for xy sequences
    """
    seq = center_sequences(seq)
    indicative_angle = np.atan2(seq[:, 0, 1], seq[:, 0, 0])
    if orientation_sensitive:
        base_orientation = (np.pi / 4) * np.floor(
            (indicative_angle + np.pi / 8) / (np.pi / 4)
        )
        delta = base_orientation - indicative_angle
    else:
        delta = -indicative_angle
    cosd = np.cos(delta)[..., None]
    sind = np.sin(delta)[..., None]
    xs = slice(None), slice(None), 0
    ys = slice(None), slice(None), 1
    new_x = seq[xs] * cosd - seq[ys] * sind
    new_y = seq[ys] * cosd + seq[xs] * sind
    seq[xs] = new_x
    seq[ys] = new_y
    norm = np.linalg.norm(seq, axis=(1, 2), keepdims=True)
    return seq / norm


def plot(queries, templates, idx, scores, angles):
    for n in range(len(queries)):
        template_idx = idx[n]
        plt.plot(queries[n, :, 0], queries[n, :, 1])
        plt.scatter(
            queries[n, :, 0], queries[n, :, 1], c=np.linspace(0, 1, queries.shape[1])
        )
        template = rotate(templates[template_idx], -angles[n])
        plt.scatter(
            template[:, 0],
            template[:, 1],
            c=np.linspace(0, 1, queries.shape[1]),
        )
        plt.title(
            f"id = {template_idx};{angles[n] / np.pi:.2f} pi; score={scores[n]:.5f}"
        )
        plt.savefig(f"./fig{n}.png")
        plt.cla()


def optimal_cosine_distances(queries: np.ndarray, templates: np.ndarray):
    """
    queries: np.ndarray of shape [batch, sequence_length, 2] for xy sequences
    templates: np.ndarray of shape [all_templates, sequence_length, 2] for xy sequences
    """
    q = queries[:, None, :, :]  # [batch, 1, sequence_length, 2]
    t = templates[None, :, :, :]  # [1, all_templates, sequence_length, 2]
    xs = slice(None), slice(None), slice(None), 0
    ys = slice(None), slice(None), slice(None), 1

    # sum over sequence_length of (qx * tx + qy * ty) a.k.a. a dot product
    a = np.einsum("btlx,btlx->bt", q, t)

    # sum over sequence_length of (qx * ty - qy * tx) a.k.a. a 2d cross product
    b = np.sum(q[xs] * t[ys] - q[ys] * t[xs], axis=-1)

    angles: np.ndarray = np.atan2(b, a)
    # why is clipping suddenly killing the quality?
    # i thought i have all possible angles and orientations
    # angles = np.clip(angles, -np.pi/4, np.pi/4)
    dot: np.ndarray = a * np.cos(angles) + b * np.sin(angles)
    return dot, angles


def recognize(queries: np.ndarray, templates: np.ndarray):
    """
    queries: np.ndarray of shape [batch, sequence_length, 2] for xy sequences
    templates: np.ndarray of shape [all_templates, sequence_length, 2] for xy sequences
    """
    assert queries.ndim == 3 and templates.ndim == 3
    assert queries.shape[2] == 2 and templates.shape[2] == 2

    scores, angles = optimal_cosine_distances(
        vectorize(queries, True), vectorize(templates, True)
    )
    idxs = np.argmax(scores, -1)
    # plot(vectorize(queries, True), vectorize(templates, True), idxs, scores[np.arange(scores.shape[0]), idxs], angles[np.arange(scores.shape[0]), idxs])
    return (
        idxs,
        scores[np.arange(scores.shape[0]), idxs],
    )
