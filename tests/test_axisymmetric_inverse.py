from __future__ import annotations

import numpy as np

from waxs_cake import conjugate_gradient_tikhonov


class MatrixOperator:
    def __init__(self, matrix: np.ndarray, object_shape: tuple[int, ...]) -> None:
        self.matrix = np.asarray(matrix, dtype=np.complex128)
        self.object_shape = object_shape
        self.data_shape = (self.matrix.shape[0],)

    def forward(self, values):
        return self.matrix @ np.asarray(values).ravel()

    def adjoint_euclidean(self, values):
        return (self.matrix.conj().T @ np.asarray(values).ravel()).reshape(
            self.object_shape
        )


def test_conjugate_gradient_tikhonov_matches_dense_ridge_solution() -> None:
    rng = np.random.default_rng(87)
    matrix = rng.normal(size=(14, 7)) + 1j * rng.normal(size=(14, 7))
    truth = rng.normal(size=7) + 1j * rng.normal(size=7)
    data = matrix @ truth
    regularization = 0.03
    operator = MatrixOperator(matrix, (7,))
    result = conjugate_gradient_tikhonov(
        operator,
        data,
        regularization=regularization,
        max_iterations=20,
        relative_tolerance=1e-12,
        truth=truth,
    )
    expected = np.linalg.solve(
        matrix.conj().T @ matrix + regularization * np.eye(matrix.shape[1]),
        matrix.conj().T @ data,
    )

    assert result.converged
    assert result.iterations <= matrix.shape[1] + 1
    assert np.allclose(result.reconstruction, expected, rtol=1e-11, atol=1e-11)
    assert result.history[-1]["relative_normal_residual"] < 1e-12
    assert result.working_set_bytes > 0


def test_conjugate_gradient_history_reduces_objective() -> None:
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(10, 5)) + 1j * rng.normal(size=(10, 5))
    operator = MatrixOperator(matrix, (5,))
    data = rng.normal(size=10) + 1j * rng.normal(size=10)
    result = conjugate_gradient_tikhonov(
        operator,
        data,
        regularization=0.1,
        max_iterations=10,
        relative_tolerance=1e-12,
    )
    objectives = np.array([entry["objective"] for entry in result.history])
    assert objectives[-1] < objectives[0]
    assert np.all(np.diff(objectives) <= 1e-10)
