import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class SPDSolver:
    """This class solves the equation:

        (Prec_x + H.T @ Prec_eps_r @ H) X = (H.T @ Prec_eps_r @ innovation)
                               Prec_eps_r = inv(Cov_r + Cov_eps)

    The key observation is that the left-hand-side is sparse, pos.def and
    updated in every MDA-like iteration, in each iteration the LHS is:

        1. (Prec_x + H_1.T @ Prec_eps_r_1 @ H_1)
        2. (Prec_x + H_1.T @ Prec_eps_r_1 @ H_1 + H_2.T @ Prec_eps_r_2 @ H_2)
        3. etc

    Two ideas for solving this equation:

        (a) Use conjugate gradients, but never explicitly form the LHS.
            Prefer many matvec operations over forming matrix (matmat).
            This is much faster, and saves storage.
        (b) Use Cholesky factorization of the sparse matrix
            (re-computing permutation and factorization was found to be
             faster than low-rank update)

    On a problem with 5000 parameters, 2500 responses and 250 realizations:
        "dense" takes 2 seconds, "cholesky" 3 seconds and "cg" 17 seconds

    Examples
    --------
    >>> import scipy as sp
    >>> Prec_x = sp.sparse.diags([1, 2, 3])
    >>> H = sp.sparse.csc_array([[1, 2, 3], [0, 2, 1]])
    >>> Prec_eps_r = np.array([0.1, 0.2])
    >>> b = sp.sparse.csc_array([[5, 1, 3], [4, 2, 3]]).T

    >>> solver = SPDSolver(Prec_x=Prec_x, solver="dense")
    >>> solver.add(H=H, Prec_eps_r=Prec_eps_r)
    >>> solver.solve(b)
    array([[ 4.44615385,  3.47384615],
           [-0.1       ,  0.28      ],
           [ 0.43076923,  0.40923077]])

    >>> solver = SPDSolver(Prec_x=Prec_x, solver="cg")
    >>> solver.add(H=H, Prec_eps_r=Prec_eps_r)
    >>> solver.solve(b)
    array([[ 4.44615385,  3.47384615],
           [-0.1       ,  0.28      ],
           [ 0.43076923,  0.40923077]])

    >>> solver = SPDSolver(Prec_x=Prec_x, solver="cholesky")
    >>> solver.add(H=H, Prec_eps_r=Prec_eps_r)
    >>> solver.solve(b)
    array([[ 4.44615385,  3.47384615],
           [-0.1       ,  0.28      ],
           [ 0.43076923,  0.40923077]])
    """

    def __init__(
        self,
        *,
        Prec_x: npt.NDArray[np.floating] | sp.sparse.sparray,
        solver: str = "dense",
        solver_options: dict[str, object] | None = None,
    ) -> None:
        assert solver in ("dense", "cg", "cholesky")
        self.solver = solver
        self.solver_options = solver_options or dict()

        Prec_x = sp.sparse.csc_array(Prec_x)

        # Depending on the solver, we keep different state
        if self.solver == "cholesky":
            from sksparse.cholmod import cholesky  # noqa: PLC0415

            # ordering_method: one of "natural", "amd", "metis",
            # "nesdis", "colamd", "default" and "best".
            # "natural" means no permutation.
            self.lhs = Prec_x.copy()
            self.lhs_factor = cholesky(self.lhs, **self.solver_options)
        elif self.solver == "cg":
            self.first = Prec_x.copy()
            self.rest: list[tuple[Any, Any]] = []
        else:
            # Here 'first' refers to the 'Prec_x' term in the LHS,
            # while 'rest' refers to the low-rank updates
            self.first = Prec_x.copy()
            self.rest = []

    def left_hand_side(self) -> npt.NDArray[np.floating] | sp.sparse.sparray:
        """Returns the left hand side, which is the posterior precision matrix."""
        # Depending on the solver, we keep different state
        if self.solver == "cholesky":
            return self.lhs.copy()
        LHS = self.first.copy()
        for H, Prec_eps_r in self.rest:
            if Prec_eps_r.ndim == 1:
                LHS += (H.T * Prec_eps_r) @ H
            else:
                LHS += H.T @ Prec_eps_r @ H
        return LHS

    def add(
        self,
        *,
        H: npt.NDArray[np.floating] | sp.sparse.sparray,
        Prec_eps_r: npt.NDArray[np.floating] | sp.sparse.sparray,
    ) -> None:
        """Add H.T @ Prec_eps_r @ H to the left-hand side of the equation."""
        if self.solver == "cholesky":
            from sksparse.cholmod import cholesky  # noqa: PLC0415

            # Update the cholesky factorization of the LHS, adding C @ C.T.
            # Let L @ L.T = Prec_eps_r, then H.T @ Prec_eps_r @ H =
            # H.T @ L @ L.T @ H = (H.T @ L) @ (H.T @ L).T
            if Prec_eps_r.ndim == 1:
                L = np.sqrt(Prec_eps_r)
                C = sp.sparse.csc_array(H.T * L)
            elif Prec_eps_r.ndim == 2 and isinstance(Prec_eps_r, np.ndarray):
                L = np.linalg.cholesky(Prec_eps_r)
                C = sp.sparse.csc_array(H.T @ L)
            elif Prec_eps_r.ndim == 2 and isinstance(Prec_eps_r, sp.sparse.sparray):
                # TODO: Use sparse cholesky instead
                L = np.linalg.cholesky(Prec_eps_r.todense())
                C = sp.sparse.csc_array(H.T @ L)
            else:
                raise ValueError("Unrecognized type")

            # self.lhs.update_inplace(C =C)
            self.lhs += C @ C.T
            self.lhs_factor = cholesky(self.lhs, **self.solver_options)

        # Store the matrices
        else:
            self.rest.append((H.copy(), Prec_eps_r.copy()))

    def solve(
        self, b: npt.NDArray[np.floating] | sp.sparse.sparray
    ) -> npt.NDArray[np.floating]:
        """Solve (Prec_x + H.T @ Prec_eps_r @ H + ...) X = b for unknown X."""

        if self.solver == "dense":
            return self._solve_dense(b)
        if self.solver == "cg":
            return self._solve_cg(b)
        if self.solver == "cholesky":
            return self._solve_cholesky(b)
        raise ValueError(f"Unknown solver: {self.solver=}")

    def _solve_dense(
        self, b: npt.NDArray[np.floating] | sp.sparse.sparray
    ) -> npt.NDArray[np.floating]:
        """Solve by forming dense matrices. Consumes a lot of memory, but
        is surprisingly fast for small problems."""
        LHS = self.first.copy()
        for H, Prec_eps_r in self.rest:
            if Prec_eps_r.ndim == 1:
                LHS += (H.T * Prec_eps_r) @ H
            else:
                LHS += H.T @ Prec_eps_r @ H

        # Densify input for the scipy cholesky solver
        LHS = LHS.todense() if isinstance(LHS, sp.sparse.sparray) else LHS
        b = b.todense() if isinstance(b, sp.sparse.sparray) else b
        result: npt.NDArray[np.floating] = sp.linalg.solve(
            LHS, b, assume_a="pos", overwrite_a=True
        )
        return result

    def _solve_cholesky(
        self, b: npt.NDArray[np.floating] | sp.sparse.sparray
    ) -> npt.NDArray[np.floating]:
        """Solve using sparse Cholesky factorization."""
        b = sp.sparse.csc_array(b)
        result: npt.NDArray[np.floating] = sp.sparse.csc_array(
            self.lhs_factor.solve_A(b)
        ).todense()
        return result

    def _solve_cg(
        self, b: npt.NDArray[np.floating] | sp.sparse.sparray
    ) -> npt.NDArray[np.floating]:
        """Solve using conjugate gradients on each column in the
        right-hand-side b."""
        assert b.ndim == 2
        b = b.todense() if isinstance(b, sp.sparse.sparray) else b

        def matvec_A(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            """Implement operator A @ v without forming A."""
            result: npt.NDArray[np.floating] = self.first @ v
            for H, Prec_eps_r in self.rest:
                if Prec_eps_r.ndim == 1:
                    # Its important to apply product right-to-left
                    result += (H.T * Prec_eps_r) @ (H @ v)
                else:
                    result += H.T @ (Prec_eps_r @ (H @ v))

            return result

        def matvec_M(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            """Implement M ~= inv(A) without forming M."""
            # using diagonal only seems faster and better than e.g.
            # cholesky factoring Prec_x
            diag: npt.NDArray[np.floating] = self.first.diagonal().copy()
            for H, Prec_eps_r in self.rest:
                if Prec_eps_r.ndim == 1:
                    # Equivalent to: ((H.T * Prec_eps_r) @ H).diagonal()
                    diag += (Prec_eps_r[:, None] * H**2).sum(axis=0)
                else:
                    # Equivalent to: ((H.T @ Prec_eps_r) @ H).diagonal()
                    diag += (H * (Prec_eps_r @ H)).sum(axis=0)

            preconditioned: npt.NDArray[np.floating] = v / diag
            return preconditioned

        n_params, n_samples = self.first.shape[0], b.shape[1]
        A = sp.sparse.linalg.LinearOperator(shape=(n_params, n_params), matvec=matvec_A)
        M = sp.sparse.linalg.LinearOperator(shape=(n_params, n_params), matvec=matvec_M)

        X = np.zeros(shape=(n_params, n_samples))  # answer
        x0 = np.zeros(n_params)
        cg_iterations = []  # Iteration counters per column of b
        # TODO: log CG errors?

        for i in range(n_samples):
            # Count number of iters to solve this column of b
            iters_i = 0

            def callback(xk: npt.NDArray[np.floating]) -> None:
                nonlocal iters_i
                iters_i = iters_i + 1

            # Relevant solver options: rtol=1e-05, atol=0.0, maxiter=None
            x, info = sp.sparse.linalg.cg(
                A=A, b=b[:, i], M=M, callback=callback, x0=x0, **self.solver_options
            )
            X[:, i] = x
            cg_iterations.append(iters_i)  # Store iteration count

        log.info("Ran conjugate gradients. Iterations per right-hand side:")
        log.info(f"mean={np.mean(cg_iterations):.1f} std={np.std(cg_iterations):.2f}")
        return X


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
        ]
    )
