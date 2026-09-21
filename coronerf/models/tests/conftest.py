from types import SimpleNamespace
import pytest


@pytest.fixture
def pipeline():
    cfg = SimpleNamespace(
        model={
            "dtype": "float",
            "strict_types": False,
            "use_sh_encoding": False,
            "sh_degree": 2,
        },
        DensityHashMLP_kwargs={
            "backend": "torch",
            "backend_fallback": True,
            "in_dim": 3,
            "n_levels": 2,
            "n_features_per_level": 2,
            "log2_hashmap_size": 5,
            "base_resolution": 4,
            "finest_resolution": 8,
            "decoder_hidden_dim": 8,
            "decoder_num_hidden": 1,
            "strict_types": False,
            "use_raw_xyz": False,
            "use_sh_encoding": False,
            "sh_degree": 2,
        },
    )
    return SimpleNamespace(cfg=cfg)