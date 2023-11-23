import numpy as np
import pytest

from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_exact_lstsq,
    inversion_exact_naive,
    inversion_exact_rescaled,
    inversion_exact_subspace_woodbury,
    inversion_rescaled_subspace,
    inversion_subspace,
    normalize_alpha,
)

# If `truncation` is 1.0, then all of these produce the same result
EXACT_INVERSIONS = [
    inversion_exact_naive,
    inversion_exact_cholesky,
    inversion_exact_rescaled,
    inversion_exact_lstsq,
    inversion_exact_subspace_woodbury,
    inversion_rescaled_subspace,
]

# Even when `truncation` is 1.0, these functions approximate the solution
# when ensemble_members <= num_outputs
APPROX_INVERSIONS = [
    inversion_subspace,
]

# The functions that make use of the truncation parameter
TRUNCATION_INVERSIONS = [
    inversion_exact_rescaled,
    inversion_subspace,
    inversion_rescaled_subspace,
]


class TestEsmdaInversion:
    # Functions that take the parameter return_K
    @pytest.mark.parametrize(
        "function",
        [inversion_exact_cholesky, inversion_subspace],
    )
    @pytest.mark.parametrize("ensemble_members", [5, 10, 15])
    @pytest.mark.parametrize("num_outputs", [5, 10, 15])
    @pytest.mark.parametrize("num_inputs", [5, 10, 15])
    def test_that_returning_T_is_equivalent_to_full_computation(
        self, function, ensemble_members, num_outputs, num_inputs
    ):
        # ensemble_members, num_outputs, num_inputs = 5, 10, 15

        np.random.seed(ensemble_members * 100 + num_outputs * 10 + num_inputs)

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = 3

        # Create observations
        D = np.random.randn(num_outputs, ensemble_members)
        Y = np.random.randn(num_outputs, ensemble_members)
        X = np.random.randn(num_inputs, ensemble_members)

        # Test both with and without X / return_T
        ans = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        T = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=None, return_T=True)

        assert np.allclose(X + ans, X + X @ T)

    @pytest.mark.parametrize("length", list(range(1, 101, 5)))
    def test_that_the_sum_of_normalize_alpha_is_one(self, length):
        # Generate random array
        rng = np.random.default_rng(length)
        alpha = np.exp(rng.normal(size=length))

        # Test the defining property of the function
        assert np.isclose(np.sum(1 / normalize_alpha(alpha)), 1)

    @pytest.mark.parametrize(
        "function",
        EXACT_INVERSIONS,
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

    @pytest.mark.parametrize(
        "function", EXACT_INVERSIONS + [inversion_rescaled_subspace]
    )
    @pytest.mark.parametrize("ensemble_members", [5, 10, 25, 50])
    @pytest.mark.parametrize("num_outputs", [5, 10, 25, 50])
    @pytest.mark.parametrize("num_inputs", [5, 10, 25, 50])
    def test_that_exact_inverions_are_equal_with_few_ensemble_members(
        self, function, ensemble_members, num_outputs, num_inputs
    ):
        """With few ensemble members, every exact inversion method should be equal."""
        np.random.seed(ensemble_members * 100 + num_outputs * 10 + num_inputs)

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = 3

        # Create observations
        D = np.random.randn(num_outputs, ensemble_members)
        Y = np.random.randn(num_outputs, ensemble_members)
        X = np.random.randn(num_inputs, ensemble_members)

        # All non-subspace methods
        K1 = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K2 = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X, truncation=1.0)
        assert np.allclose(K1, K2)

    @pytest.mark.parametrize("function", APPROX_INVERSIONS)
    @pytest.mark.parametrize("num_outputs", [10])
    def test_that_approximate_inversions_do_not_compute_exact_answer_with_few_members(
        self, function, num_outputs
    ):
        """With few ensemble members, the approximate methods should not return
        the exact answer, even when truncation=1.0."""
        np.random.seed(num_outputs)

        num_inputs = num_outputs
        ensemble_members = num_outputs // 2

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = 3

        # Create observations
        D = np.random.randn(num_outputs, ensemble_members)
        Y = np.random.randn(num_outputs, ensemble_members)
        X = np.random.randn(num_inputs, ensemble_members)

        # All non-subspace methods
        K1 = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)
        K2 = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X, truncation=1.0)
        assert not np.allclose(K1, K2)

    @pytest.mark.parametrize(
        "function",
        EXACT_INVERSIONS + APPROX_INVERSIONS,
    )
    @pytest.mark.parametrize(
        "num_outputs,num_emsemble",
        [(10, 11), (10, 20), (10, 100), (100, 101), (100, 200), (100, 500)],
    )
    def test_that_all_inversions_all_equal_with_many_ensemble_members(
        self, function, num_outputs, num_emsemble
    ):
        """As long as ensemble_members > num_outputs and truncation=1.0, every
        inversion method should return the same result, whether it's an exact
        method or an approximate method."""
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

        K1 = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X, truncation=1.0)
        K2 = function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X, truncation=1.0)

        assert np.allclose(K1, K2)

    @pytest.mark.parametrize("function", TRUNCATION_INVERSIONS)
    @pytest.mark.parametrize("ensemble_over_outputs", [2, 1, 0.5])
    def test_that_approximations_get_better_when_truncation_is_increased(
        self, function, ensemble_over_outputs
    ):
        num_outputs = 16
        num_emsemble = int(ensemble_over_outputs * num_outputs)
        num_inputs = 8
        np.random.seed(num_outputs + num_emsemble + num_inputs)

        # Create positive symmetric definite covariance C_D
        E = np.random.randn(num_outputs, num_outputs)
        C_D = E.T @ E

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, num_emsemble)
        Y = np.random.randn(num_outputs, num_emsemble)
        X = np.random.randn(num_inputs, num_emsemble)

        # Exact answer
        exact = inversion_exact_naive(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)

        # Approximate answers
        truncations = np.linspace(1e-8, 1, num=64)
        approximations = [
            function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X, truncation=trunaction)
            for trunaction in truncations
        ]

        norms = np.array(
            [np.linalg.norm(approx_i - exact) for approx_i in approximations]
        )

        # As we increase `truncation`, the difference between the approximation
        # and the true answer should decrease
        assert np.all(np.diff(norms) <= 0)

        # Test that there is significant difference between endpoints
        assert abs(norms[0] - norms[-1]) > 1

    @pytest.mark.parametrize("ratio_ensemble_members_over_outputs", [0.5, 1, 2])
    @pytest.mark.parametrize("num_outputs", [10, 50, 100])
    @pytest.mark.parametrize(
        "function",
        EXACT_INVERSIONS + APPROX_INVERSIONS,
    )
    def test_that_inversion_methods_work_with_covariance_matrix_and_variance_vector(
        self, ratio_ensemble_members_over_outputs, num_outputs, function
    ):
        """The result should be the same whether C_D is diagonal or not."""
        ensemble_members = int(num_outputs / ratio_ensemble_members_over_outputs)
        num_inputs = num_outputs
        np.random.seed(num_outputs + ensemble_members)

        # Diagonal covariance matrix
        C_D_2D = np.diag(np.exp(np.random.randn(num_outputs)))
        C_D_diag = np.diag(C_D_2D)

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, ensemble_members)
        Y = np.random.randn(num_outputs, ensemble_members)
        X = np.random.randn(num_inputs, ensemble_members)

        assert C_D_2D.ndim == 2
        assert C_D_diag.ndim == 1

        result_diag = function(alpha=alpha, C_D=C_D_diag, D=D, Y=Y, X=X)
        result_2D = function(alpha=alpha, C_D=C_D_2D, D=D, Y=Y, X=X)

        assert np.allclose(result_diag, result_2D)

    @pytest.mark.parametrize(
        "function",
        EXACT_INVERSIONS + APPROX_INVERSIONS,
    )
    def test_that_inversion_methods_do_do_not_mutate_input_args(self, function):
        """Inversion functions should not mutate input arguments."""
        num_outputs, num_inputs, ensemble_members = 100, 50, 10

        np.random.seed(42)

        # Diagonal covariance matrix
        C_D = np.diag(np.exp(np.random.randn(num_outputs)))

        # Set alpha to something other than 1 to test that it works
        alpha = np.exp(np.random.randn())

        # Create observations
        D = np.random.randn(num_outputs, ensemble_members)
        Y = np.random.randn(num_outputs, ensemble_members)
        X = np.random.randn(num_inputs, ensemble_members)

        args = [alpha, C_D, D, Y, X]
        args_copy = [np.copy(arg) for arg in args]

        function(alpha=alpha, C_D=C_D, D=D, Y=Y, X=X)

        for arg, arg_copy in zip(args, args_copy):
            assert np.allclose(arg, arg_copy)


def test_timing(num_outputs=100, num_inputs=50, ensemble_members=25):
    E = np.random.randn(num_outputs, num_outputs)
    C_D = E.T @ E
    C_D = np.diag(np.exp(np.random.randn(num_outputs)))  # Diagonal covariance matrix
    C_D_diag = np.diag(C_D)
    assert C_D_diag.ndim == 1
    assert C_D.ndim == 2

    # Set alpha to something other than 1 to test that it works
    alpha = 3

    # Create observations
    D = np.random.randn(num_outputs, ensemble_members)
    Y = np.random.randn(num_outputs, ensemble_members)
    X = np.random.randn(num_inputs, ensemble_members)

    exact_inversion_funcs = [
        inversion_exact_naive,
        inversion_exact_cholesky,
        # inversion_exact_rescaled,
        # inversion_lstsq,
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
        inversion_exact_subspace_woodbury,
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
            "-k test_that_returning_K_is_equivalent_to_full_computation",
        ]
    )
