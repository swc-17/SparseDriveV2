import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from navsim.planning.training.dataset import load_feature_target_from_pickle

K_PATH = 1024
K_VELOCITY = 256
CACHE_PATH = "exp/data_cache_navtrain"
VIS_DIR = "vis"
CKPT_DIR = "ckpt/kmeans"
DT = 0.5

LOG_NAMES = os.listdir(CACHE_PATH)


def interp1d_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    y = np.interp(x, xp, fp)

    m_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
    left_mask = x < xp[0]
    y[left_mask] = fp[0] + m_left * (x[left_mask] - xp[0])

    m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    right_mask = x > xp[-1]
    y[right_mask] = fp[-1] + m_right * (x[right_mask] - xp[-1])

    return y


def interp_trajectory(
    path_cluster: np.ndarray,
    velocity_cluster: np.ndarray,
    interp_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = np.interp,
) -> Tuple[np.ndarray, np.ndarray]:
    num_velocity = velocity_cluster.shape[1]
    trajectory = np.zeros((K_PATH, K_VELOCITY, num_velocity, 3))
    trajectory_mask = np.ones((K_PATH, K_VELOCITY, num_velocity))

    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            path = path_cluster[i]
            velocity = velocity_cluster[j]

            target_distance = np.cumsum(velocity * DT, axis=0)
            pad_path = np.concatenate([np.zeros((1, 3)), path], axis=0)
            distance = np.linalg.norm(pad_path[1:, :2] - pad_path[:-1, :2], axis=-1).cumsum(axis=0)
            distance = np.concatenate([np.zeros((1,)), distance], axis=0)
            interp_traj = np.array(
                [
                    interp_func(target_distance, distance, pad_path[:, 0]),
                    interp_func(target_distance, distance, pad_path[:, 1]),
                    interp_func(target_distance, distance, pad_path[:, 2]),
                ]
            ).T
            interp_traj[:, 2] = (interp_traj[:, 2] + np.pi) % (2 * np.pi) - np.pi
            trajectory[i, j] = interp_traj

            max_dist = distance[-1]
            valid = target_distance <= max_dist
            trajectory_mask[i, j, ~valid] = 0.0

    return trajectory, trajectory_mask


def load_one(cache_path: str, log_name: str, token: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
    data_path = os.path.join(cache_path, log_name, token, "sparsedrive_target.gz")
    data = load_feature_target_from_pickle(data_path)
    assert "path" in data

    if data["path_mask"].all():
        path = np.array(data["path"], copy=True)
        path[:, 2] = (path[:, 2] + np.pi) % (2 * np.pi) - np.pi
    else:
        path = None
    velocity = data["velocity"]
    return path, velocity


def load_all_parallel(
    cache_path: str, log_names: List[str], max_workers: int = 64
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    paths: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    tasks = []
    for log_name in log_names:
        tokens = os.listdir(os.path.join(cache_path, log_name))
        for token in tokens:
            tasks.append((log_name, token))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_one, cache_path, log_name, token)
            for (log_name, token) in tasks
        ]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            path, velocity = fut.result()
            if path is not None:
                paths.append(path)
            velocities.append(velocity)

    return paths, velocities


def visualize(path_cluster: np.ndarray, velocity_cluster: np.ndarray, trajectory: np.ndarray) -> None:
    os.makedirs(VIS_DIR, exist_ok=True)

    for j in range(K_PATH):
        plt.plot(path_cluster[j, :, 0], path_cluster[j, :, 1])
    plt.savefig(f"{VIS_DIR}/path_{K_PATH}.png", bbox_inches="tight")
    plt.close()

    num_velocity = velocity_cluster.shape[1]
    colors = plt.cm.Spectral(np.linspace(0, 1, num_velocity))
    x = np.arange(K_VELOCITY)
    plt.figure(figsize=(10, 4))

    plt.bar(x, velocity_cluster[:, 0], color=colors[0], label="0-0.5 s")
    bottom = velocity_cluster[:, 0].copy()
    for i in range(1, num_velocity):
        plt.bar(
            x,
            velocity_cluster[:, i],
            bottom=bottom,
            color=colors[i],
            label=f"{i * DT:.1f}-{(i + 1) * DT:.1f} s",
        )
        bottom += velocity_cluster[:, i]
    plt.xlabel("sequence index")
    plt.ylabel("speed (m/s)")
    plt.title("Stacked speed histogram")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/velocity_{K_VELOCITY}.png", bbox_inches="tight")
    plt.close()

    for i in range(K_PATH):
        for j in range(K_VELOCITY):
            plt.plot(trajectory[i, j, :, 0], trajectory[i, j, :, 1])
    plt.savefig(f"{VIS_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.png", bbox_inches="tight")
    plt.close()

def save_outputs(
    path_cluster: np.ndarray,
    velocity_cluster: np.ndarray,
    trajectory: np.ndarray,
    trajectory_mask: np.ndarray,
) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    np.save(f"{CKPT_DIR}/path_{K_PATH}.npy", path_cluster)
    np.save(f"{CKPT_DIR}/velocity_{K_VELOCITY}.npy", velocity_cluster)
    np.savez(
        f"{CKPT_DIR}/trajectory_{K_PATH}_{K_VELOCITY}.npz",
        trajectory=trajectory,
        trajectory_mask=trajectory_mask,
    )


def main():
    paths, velocities = load_all_parallel(CACHE_PATH, LOG_NAMES, max_workers=64)

    print("total path / velocity length:", len(paths), len(velocities))

    num_pts = paths[0].shape[0]
    paths_flatten = np.stack(paths).reshape(len(paths), -1)
    path_cluster = KMeans(n_clusters=K_PATH).fit(paths_flatten).cluster_centers_
    path_cluster = path_cluster.reshape(K_PATH, num_pts, 3)
    path_cluster[:, :, 2] = (path_cluster[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

    velocities = np.stack(velocities)
    velocity_cluster = KMeans(n_clusters=K_VELOCITY).fit(velocities).cluster_centers_

    trajectory, trajectory_mask = interp_trajectory(path_cluster, velocity_cluster, interp_func=interp1d_extrap)

    visualize(path_cluster, velocity_cluster, trajectory)
    save_outputs(path_cluster, velocity_cluster, trajectory, trajectory_mask)

if __name__ == "__main__":
    main()