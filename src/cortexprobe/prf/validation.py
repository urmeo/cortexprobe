"""Grouped scoring."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..arrays import FloatArray, IntArray, unit_responses
from ..config import FitConfig
from ..geometry import Grid
from .fit import N_PARAMETERS, PRFFitter, UnitFit, _r_squared
from .model import GaussianReceptiveField, predict


@dataclass(frozen=True)
class CrossValidatedFit:
    """Validation results."""

    fit: UnitFit
    fold_r2: tuple[float, ...]
    n_skipped_folds: int = 0

    @property
    def cv_r2(self) -> float:
        """Finite mean."""
        scores = [score for score in self.fold_r2 if np.isfinite(score)]
        if not scores:
            return float("nan")
        scale = float(max(abs(score) for score in scores))
        return float(np.mean(np.asarray(scores) / scale)) * scale if scale else 0.0

    @property
    def cv_r2_sd(self) -> float:
        """Sample deviation."""
        scores = [score for score in self.fold_r2 if np.isfinite(score)]
        if len(scores) < 2:
            return float("nan")
        scale = float(max(abs(score) for score in scores))
        return float(np.std(np.asarray(scores) / scale, ddof=1)) * scale if scale else 0.0

    @property
    def n_folds(self) -> int:
        """Retained folds."""
        return len(self.fold_r2)

    @property
    def n_total_folds(self) -> int:
        """Requested folds."""
        return self.n_folds + self.n_skipped_folds

    @property
    def n_valid_folds(self) -> int:
        """Finite folds."""
        return sum(1 for score in self.fold_r2 if np.isfinite(score))

    @property
    def validation_complete(self) -> bool:
        return self.n_total_folds > 0 and self.n_valid_folds == self.n_total_folds

    @property
    def optimism(self) -> float:
        return self.fit.r2 - self.cv_r2


class LeaveOneGroupOut:
    """Whole-group folds."""

    def __init__(self, groups: Sequence[int]) -> None:
        self._groups = np.array(groups, copy=True)
        if self._groups.ndim != 1:
            raise ValueError("groups must be one-dimensional")
        try:
            finite = bool(np.isfinite(self._groups).all())
        except TypeError:
            finite = False
        if not finite:
            raise ValueError("groups must be finite numeric labels")
        self._unique: np.ndarray = np.unique(self._groups)
        if len(self._unique) < 2:
            raise ValueError("cross-validation needs at least two sweep groups")
        self._groups.setflags(write=False)
        self._unique.setflags(write=False)

    @property
    def groups(self) -> np.ndarray:
        return self._groups.view()

    @property
    def unique(self) -> np.ndarray:
        return self._unique.view()

    def __len__(self) -> int:
        return len(self.unique)

    def splits(self) -> Iterator[tuple[IntArray, IntArray]]:
        for held_out in self.unique:
            test = np.flatnonzero(self.groups == held_out)
            train = np.flatnonzero(self.groups != held_out)
            yield train, test


class CrossValidator:
    """Grouped validation."""

    def __init__(
        self,
        grid: Grid,
        apertures: FloatArray,
        groups: Sequence[int],
        config: FitConfig,
    ) -> None:
        self.grid = grid
        apertures = np.asarray(apertures, dtype=np.float64)
        self.config = config
        self.splitter = LeaveOneGroupOut(groups)
        if apertures.ndim != 3:
            raise ValueError("apertures must have shape (frames, height, width)")
        if len(self.splitter.groups) != len(apertures):
            raise ValueError("groups must have one group per stimulus frame")

        self._full = PRFFitter(grid, apertures, config)
        self._folds: list[tuple[IntArray, IntArray, PRFFitter]] = []
        for train, test in self.splitter.splits():
            if len(train) <= N_PARAMETERS:
                continue
            self._folds.append((train, test, PRFFitter(grid, self.apertures[train], config)))
        if not self._folds:
            raise ValueError("no fold retains enough frames to fit a pRF")

    @property
    def apertures(self) -> FloatArray:
        """Stimulus snapshot."""
        return self._full.apertures

    def validate_unit(self, response: FloatArray) -> CrossValidatedFit:
        response = np.asarray(response, dtype=np.float64)
        full_fit = self._full.fit_unit(response)

        scores: list[float] = []
        for train, test, fitter in self._folds:
            train_fit = fitter.fit_unit(response[train])
            scores.append(self._score_held_out(train_fit, test, response))
        return CrossValidatedFit(
            fit=full_fit,
            fold_r2=tuple(scores),
            n_skipped_folds=len(self.splitter) - len(self._folds),
        )

    def _score_held_out(self, train_fit: UnitFit, test: IntArray, response: FloatArray) -> float:
        if not np.isfinite([train_fit.x0, train_fit.y0, train_fit.sigma]).all():
            return float("nan")
        field = GaussianReceptiveField(train_fit.x0, train_fit.y0, train_fit.sigma)
        prediction = predict(field.weights(self.grid), self.apertures[test])
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = train_fit.beta * prediction + train_fit.baseline
            if not np.isfinite(fitted).all():
                scale = max(abs(train_fit.beta), abs(train_fit.baseline)) or 1.0
                fitted = (
                    (train_fit.beta / scale) * prediction + train_fit.baseline / scale
                ) * scale
        return _r_squared(response[test], fitted)

    def validate_all(self, activations: FloatArray) -> list[CrossValidatedFit]:
        return [
            self.validate_unit(response)
            for response in unit_responses(activations, len(self.apertures))
        ]
