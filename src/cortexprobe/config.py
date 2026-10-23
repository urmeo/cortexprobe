"""Validated settings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Optional, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T", bound="ConfigBase")


class ConfigError(ValueError):
    """Invalid settings."""


def _require_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")


def _finite_float(value: float, name: str, error_type: type[ValueError] = ConfigError) -> float:
    try:
        valid = (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        raise error_type(f"{name} must be finite")
    return float(value) if value else 0.0


def _restore_tuples(annotation: Any, value: Any) -> Any:
    if get_origin(annotation) is not tuple:
        return value
    if not isinstance(value, (list, tuple)):
        raise TypeError("tuple fields require a list or tuple")
    element_type = (get_args(annotation) or (Any,))[0]
    return tuple(_restore_tuples(element_type, item) for item in value)


@dataclass(frozen=True)
class ConfigBase:
    """Serializable settings."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _fraction(self, name: str) -> None:
        value = _finite_float(getattr(self, name), name)
        if not 0.0 < value < 1.0:
            raise ConfigError(f"{name} must lie in (0, 1)")
        object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
        if not isinstance(payload, dict):
            raise ConfigError(f"{cls.__name__} must be an object")
        field_types = get_type_hints(cls)
        field_names = {field.name for field in fields(cls)}
        unknown_keys = set(payload) - field_names
        if unknown_keys:
            raise ConfigError(f"{cls.__name__} received unknown keys: {sorted(unknown_keys)}")
        try:
            restored = {
                name: _restore_tuples(field_types[name], value) for name, value in payload.items()
            }
        except TypeError as exc:
            raise ConfigError(f"{cls.__name__} has an invalid container field") from exc
        return cls(**restored)


@dataclass(frozen=True)
class StimulusConfig(ConfigBase):
    resolution: int = 128
    n_steps: int = 32
    bar_width_frac: float = 0.125
    directions: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    wedge_span_deg: float = 45.0
    ring_thickness_frac: float = 0.125
    max_fold_similarity: float = 0.75

    def __post_init__(self) -> None:
        _require_integer(self.resolution, "resolution")
        _require_integer(self.n_steps, "n_steps")
        if self.resolution < 8:
            raise ConfigError("resolution must be at least 8 pixels")
        if self.resolution % 2:
            raise ConfigError(
                "resolution must be even to keep the field symmetric about the origin"
            )
        if self.n_steps < 2:
            raise ConfigError("n_steps must be at least 2")
        self._fraction("bar_width_frac")
        if not isinstance(self.directions, (tuple, list)) or not self.directions:
            raise ConfigError("at least one sweep direction is required")
        for direction in self.directions:
            _require_integer(direction, "directions")
        if any(not 0 <= d < 360 for d in self.directions):
            raise ConfigError("directions must be degrees in [0, 360)")
        if len(set(self.directions)) != len(self.directions):
            raise ConfigError("directions must be unique")
        object.__setattr__(self, "directions", tuple(self.directions))
        object.__setattr__(
            self, "wedge_span_deg", _finite_float(self.wedge_span_deg, "wedge_span_deg")
        )
        if not 0.0 < self.wedge_span_deg <= 360.0:
            raise ConfigError("wedge_span_deg must lie in (0, 360]")
        self._fraction("ring_thickness_frac")
        self._fraction("max_fold_similarity")

    @property
    def n_bar_frames(self) -> int:
        return self.n_steps * len(self.directions)

    @property
    def bar_width_px(self) -> float:
        return self.bar_width_frac * self.resolution

    @property
    def ring_thickness_px(self) -> float:
        return self.ring_thickness_frac * self.resolution


@dataclass(frozen=True)
class ModelConfig(ConfigBase):
    """Reserved metadata."""

    name: str = "alexnet"
    layers: tuple[str, ...] = ("features.2", "features.5", "features.12")
    weights_seed: Optional[int] = None
    pool_to: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("model name must not be empty")
        if not isinstance(self.layers, (tuple, list)) or not self.layers:
            raise ConfigError("at least one tap layer is required")
        if any(not isinstance(layer, str) or not layer.strip() for layer in self.layers):
            raise ConfigError("tap layers must be nonempty names")
        if len(set(self.layers)) != len(self.layers):
            raise ConfigError("tap layers must be unique")
        object.__setattr__(self, "layers", tuple(self.layers))
        _require_integer(self.pool_to, "pool_to")
        if self.pool_to < 2:
            raise ConfigError("pool_to must be at least 2 to retain retinotopic structure")
        if self.weights_seed is not None:
            _require_integer(self.weights_seed, "weights_seed")
            if self.weights_seed < 0:
                raise ConfigError("weights_seed must be nonnegative")


@dataclass(frozen=True)
class FitConfig(ConfigBase):
    grid_size: int = 12
    sigma_bounds: tuple[float, float] = (1.0, 20.0)
    max_nfev: int = 200
    r2_threshold: float = 0.2

    def __post_init__(self) -> None:
        _require_integer(self.grid_size, "grid_size")
        _require_integer(self.max_nfev, "max_nfev")
        if self.grid_size < 3:
            raise ConfigError("grid_size must be at least 3")
        if not isinstance(self.sigma_bounds, (tuple, list)) or len(self.sigma_bounds) != 2:
            raise ConfigError("sigma_bounds must contain exactly two bounds")
        low, high = self.sigma_bounds
        low = _finite_float(low, "lower sigma bound")
        high = _finite_float(high, "upper sigma bound")
        if low < 1.0:
            raise ConfigError(
                "lower sigma bound must be at least 1 pixel; below the pixel pitch a Gaussian "
                "is under-sampled and loses its unit-volume normalisation"
            )
        if high <= low:
            raise ConfigError("sigma_bounds must be increasing")
        object.__setattr__(self, "sigma_bounds", (low, high))
        if self.max_nfev < 1:
            raise ConfigError("max_nfev must be positive")
        object.__setattr__(self, "r2_threshold", _finite_float(self.r2_threshold, "r2_threshold"))
        if not 0.0 <= self.r2_threshold <= 1.0:
            raise ConfigError("r2_threshold must lie in [0, 1]")


@dataclass(frozen=True)
class RunConfig(ConfigBase):
    stimulus: StimulusConfig = StimulusConfig()
    model: ModelConfig = ModelConfig()
    fit: FitConfig = FitConfig()
    seed: int = 0

    _SECTIONS: ClassVar[dict[str, type[ConfigBase]]] = {
        "stimulus": StimulusConfig,
        "model": ModelConfig,
        "fit": FitConfig,
    }

    def __post_init__(self) -> None:
        _require_integer(self.seed, "seed")
        if self.seed < 0:
            raise ConfigError("seed must be nonnegative")
        for name, section_type in self._SECTIONS.items():
            if not isinstance(getattr(self, name), section_type):
                raise ConfigError(f"{name} must be a {section_type.__name__}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        if not isinstance(payload, dict):
            raise ConfigError("RunConfig must be an object")
        sections: dict[str, Any] = dict(payload)
        for name, section_type in cls._SECTIONS.items():
            if name in sections:
                sections[name] = section_type.from_dict(sections[name])
        return super().from_dict(sections)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
