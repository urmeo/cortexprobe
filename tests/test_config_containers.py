"""Container inputs must not become character or mapping-key layer names."""

import pytest

from cortexprobe.config import ConfigError, ModelConfig


@pytest.mark.parametrize("layers", ["conv", {"conv1": 1, "conv2": 2}])
def test_serialized_layer_container_rejects_nonsequences(layers) -> None:
    with pytest.raises(ConfigError, match="invalid container"):
        ModelConfig.from_dict({"layers": layers})


@pytest.mark.parametrize("layers", [["conv1", "conv2"], ("conv1", "conv2")])
def test_serialized_layer_sequences_keep_complete_names(layers) -> None:
    restored = ModelConfig.from_dict({"layers": layers})
    assert restored.layers == ("conv1", "conv2")
