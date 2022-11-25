import numpy as np
import numpy.typing as npt


def _compute_AA_projection(A: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """A^+A projection is necessary when the parameter matrix has fewer rows than
    columns, and when the forward model is non-linear. Section 2.4.3
    """
    _, _, vh = np.linalg.svd(A - A.mean(axis=1, keepdims=True), full_matrices=False)
    return vh.T @ vh
