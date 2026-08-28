from __future__ import annotations

import h5py
import numpy as np

from clear_lewm.datasets import sample_rows, split_episode_ids, valid_pairs


def test_valid_pairs_do_not_cross_episode(cube_dataset):
    with h5py.File(cube_dataset, "r") as dataset:
        starts, goals, episodes, steps = valid_pairs(dataset, goal_offset=2)
    assert len(starts) == 20 * 30
    assert np.all(episodes[starts] == episodes[goals])
    assert np.all(steps[goals] - steps[starts] == 2)


def test_episode_split_is_exact_and_deterministic():
    episodes = np.repeat(np.arange(20), 3)
    train_a, heldout_a = split_episode_ids(episodes, 0.2, seed=42)
    train_b, heldout_b = split_episode_ids(episodes, 0.2, seed=42)
    assert len(heldout_a) == 4
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(heldout_a, heldout_b)
    assert not np.intersect1d(train_a, heldout_a).size


def test_episode_balanced_sampling_uses_distinct_episodes():
    episode_ids = np.repeat(np.arange(10), 5)
    candidates = np.arange(len(episode_ids))
    sampled = sample_rows(
        candidates, episode_ids, num_eval=8, seed=7, mode="episode-balanced"
    )
    assert len(sampled) == 8
    assert len(np.unique(episode_ids[sampled])) == 8
