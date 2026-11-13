from __future__ import annotations

import json

import pytest

from src.config import Config


def test_default_config_is_valid():
    config = Config()

    assert config.seed == 42
    assert config.n_clients == 20
    assert config.mal_fraction == pytest.approx(0.2)
    assert config.epochs == 50
    assert config.lr == pytest.approx(0.2)
    assert config.attack_type == "random"


def test_config_to_dict_is_json_serializable(tmp_path):
    config = Config(data_path="data/framingham.csv", results_dir=tmp_path / "results")
    payload = config.to_dict()

    encoded = json.dumps(payload)
    assert "data/framingham.csv" in encoded
    assert payload["data_path"] == "data/framingham.csv"
    assert payload["results_dir"] == str(tmp_path / "results")


def test_config_save_and_load_roundtrip(tmp_path):
    config = Config(seed=7, n_clients=10, attack_type="sign_flip")
    path = config.save(tmp_path / "config.json")

    loaded = Config.load(path)

    assert loaded == config


def test_from_dict_ignores_unknown_keys():
    config = Config.from_dict({"seed": 3, "unknown_key": "ignored"})

    assert config.seed == 3


def test_ensure_results_dir(tmp_path):
    results_dir = tmp_path / "nested" / "results"
    config = Config(results_dir=results_dir)

    returned = config.ensure_results_dir()

    assert returned == results_dir
    assert results_dir.is_dir()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"val_size": 0.0},
        {"test_size": 0.8, "val_size": 0.3},
        {"n_clients": 1},
        {"mal_fraction": -0.1},
        {"mal_fraction": 1.0},
        {"epochs": 0},
        {"lr": 0.0},
        {"theta": -0.1},
        {"theta": 1.1},
        {"z0": -0.1},
        {"attack_type": "unknown"},
        {"attack_strength": -0.1},
        {"target_col": "   "},
    ],
)
def test_invalid_config_values_raise(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
