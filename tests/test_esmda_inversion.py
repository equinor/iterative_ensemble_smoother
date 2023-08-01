import numpy as np
import pytest

from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_lstsq,
    inversion_exact_naive,
    inversion_exact_rescaled,
    inversion_rescaled_subspace,
    inversion_subspace,
    inversion_subspace_woodbury,
    normalize_alpha,
)


class TestEsmdaInversion:
    @pytest.mark.parametrize("length", list(range(1, 101, 5)))
    def test_that_the_sum_of_normalize_alpha_is_one(self, length):
        # Generate random array
        rng = np.random.default_rng(length)
        alpha = np.exp(rng.normal(size=length))

        # Test the defining property of the function
        assert np.isclose(np.sum(1 / normalize_alpha(alpha)), 1)

    @pytest.mark.parametrize(
        "function",
        [
            inversion_exact_naive,
            inversion_exact_cholesky,
            inversion_exact_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
        ],
    )
    def test_exact_inversion_on_a_simple_example(self, function):
        """Test on one of the simplest cases imaginable.

        If C_D = diag([1, 1, 1]) and Y = array([[2, 0],
                                                [0, 0],
                                                [0, 0]])

        then C_DD = diag([2, 0, 0])

        and inv(C_DD + C_D) = diag([1/3, 1, 1])
        """

        # Create positive symmetric definite covariance C_D
        C_D = np.identity(3)

        # Create observations
        D = np.ones((3, 2))
        Y = np.array([[2, 0], [0, 0], [0, 0]], dtype=float)
        X = np.zeros((3, 2), dtype=float)
        np.fill_diagonal(X, 1.0)

        # inv(diag([3, 1, 1])) @ (D - Y)
        K0 = function(alpha=1, C_D=C_D, D=D, Y=Y, X=X)
        assert np.allclose(K0, np.array([[-1 / 3, 1 / 3], [1 / 3, -1 / 3], [0, 0]]))

        # Same thing, but with a diagonal covariance represented as an array
        K0 = function(alpha=1, C_D=np.diag(C_D), D=D, Y=Y, X=X)
        assert np.allclose(K0, np.array([[-1 / 3, 1 / 3], [1 / 3, -1 / 3], [0, 0]]))

    def test_that_inverions_are_equal_with_few_ensemble_members(self, k=10):
        emsemble_members = k - 1

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(k, k)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = 3

        # Create observations
        D = np.random.randn(k, emsemble_members)
        Y = np.random.randn(k, emsemble_members)
        X = np.random.randn(k, emsemble_members)

        # All non-subspace methods
        K1 = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K2 = inversion_exact_cholesky(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K3 = inversion_exact_rescaled(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K4 = inversion_lstsq(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)

        assert np.allclose(K1, K2)
        assert np.allclose(K1, K3)
        assert np.allclose(K1, K4)

    @pytest.mark.parametrize(
        "function",
        [
            # Exact inversions
            inversion_exact_naive,
            inversion_exact_cholesky,
            inversion_exact_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
            # Approximate inversions (same result as long as ensemble_members > num_outputs)
            inversion_subspace,
            inversion_rescaled_subspace,
        ],
    )
    @pytest.mark.parametrize(
        "num_outputs,num_emsemble",
        [(10, 11), (10, 20), (10, 100), (100, 101), (100, 200), (100, 500)],
    )
    def test_that_all_inversions_all_equal_with_many_ensemble_members(
        self, function, num_outputs, num_emsemble
    ):
        assert num_emsemble > num_outputs
        np.random.seed(num_outputs + num_emsemble)
        num_inputs = num_outputs

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, num_emsemble)
        Y = np.random.randn(num_outputs, num_emsemble)
        X = np.random.randn(num_inputs, num_emsemble)

        K1 = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K2 = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)

        assert np.allclose(K1, K2)

    @pytest.mark.parametrize("ratio_ensemble_members_over_outputs", [0.5, 1, 2])
    @pytest.mark.parametrize("num_outputs", [10, 50, 100])
    @pytest.mark.parametrize(
        "function",
        [
            inversion_exact_naive,
            inversion_exact_cholesky,
            inversion_exact_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
            inversion_subspace,
            inversion_rescaled_subspace,
        ],
    )
    def test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector(
        self, ratio_ensemble_members_over_outputs, num_outputs, function
    ):
        emsemble_members = int(num_outputs / ratio_ensemble_members_over_outputs)
        num_inputs = num_outputs
        np.random.seed(num_outputs + emsemble_members)

        # Diagonal covariance matrix
        C_D = np.diag(np.exp(np.random.randn(num_outputs)))
        C_D_full = np.diag(C_D)

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, emsemble_members)
        Y = np.random.randn(num_outputs, emsemble_members)
        X = np.random.randn(num_inputs, emsemble_members)

        result_diagonal = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        result_dense = function(alpha=alpha, C_D=C_D_full, D=D, Y=Y, X=X)

        assert np.allclose(result_diagonal, result_dense)

    @pytest.mark.parametrize(
        "function",
        [
            inversion_exact_naive,
            inversion_exact_cholesky,
            inversion_exact_rescaled,
            inversion_lstsq,
            inversion_subspace_woodbury,
            inversion_subspace,
            inversion_rescaled_subspace,
        ],
    )
    def test_that_inversion_methods_do_do_not_mutate_input_args(self, function):
        num_outputs, num_inputs, emsemble_members = 100, 50, 10

        np.random.seed(42)

        # Diagonal covariance matrix
        C_D = np.diag(np.exp(np.random.randn(num_outputs)))

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, emsemble_members)
        Y = np.random.randn(num_outputs, emsemble_members)
        X = np.random.randn(num_inputs, emsemble_members)

        args = [alpha, C_D, D, Y, X]
        args_copy = [np.copy(arg) for arg in args]

        function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)

        for arg, arg_copy in zip(args, args_copy):
            assert np.allclose(arg, arg_copy)


def test_timing(num_outputs=100, num_inputs=50, num_ensemble=25):
    k = num_outputs
    emsemble_members = num_ensemble

    E = np.random.randn(k, k)
    C_D = E.T @ E
    C_D = np.diag(np.exp(np.random.randn(k)))  # Diagonal covariance matrix
    C_D_diag = np.diag(C_D)
    assert C_D_diag.ndim == 1
    assert C_D.ndim == 2

    # Set alpha to something other than 1 to test that it works
    alpha = 3

    # Create observations
    D = np.random.randn(k, emsemble_members)
    Y = np.random.randn(k, emsemble_members)
    X = np.random.randn(num_inputs, emsemble_members)

    exact_inversion_funcs = [
        inversion_exact_naive,
        inversion_exact_cholesky,
        inversion_exact_rescaled,
        inversion_lstsq,
    ]

    from time import perf_counter

    print("-" * 32)

    for func in exact_inversion_funcs:
        start_time = perf_counter()
        result_matrix = func(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on dense covariance: {elapsed_time} s")

        start_time = perf_counter()
        result_vector = func(alpha=alpha, C_D=C_D_diag, D=D, Y=Y, X=X)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on diagonal covariance: {elapsed_time} s")
        assert np.allclose(result_matrix, result_vector)

        print("-" * 32)

    subspace_inversion_funcs = [
        inversion_subspace,
        inversion_subspace_woodbury,
        inversion_rescaled_subspace,
    ]

    from time import perf_counter

    for func in subspace_inversion_funcs:
        start_time = perf_counter()
        result_matrix = func(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on dense covariance: {elapsed_time} s")

        start_time = perf_counter()
        result_vector = func(alpha=alpha, C_D=C_D_diag, D=D, Y=Y, X=X)
        elapsed_time = round(perf_counter() - start_time, 4)
        print(f"Function: {func.__name__} on diagonal covariance: {elapsed_time} s")
        assert np.allclose(result_matrix, result_vector)

        print("-" * 32)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "-v",
            # "-k test_that_inversion_methods_do_do_not_mutate_input_args",
        ]
    )
