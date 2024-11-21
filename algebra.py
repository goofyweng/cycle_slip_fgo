import numpy as np
import variables

#Linear algebra functions

#inversion of Matrix M
def invert_matrix(M):
    m,n = M.shape
    assert m == n, "matrix not square"
    spectrum = np.linalg.svd(M)[1]
    largest_sv = spectrum[0]
    lowest_sv = spectrum[-1]
    assert variables.relative_numerical_threshold <= lowest_sv/largest_sv, "bad eig_1/eig_n of matrix, ie badly conditioned matrix"
    assert variables.absolute_numerical_threshold <= lowest_sv, "lower singular value too close to zero, singular matrix"
    return np.linalg.solve(M,np.eye(n))

#assertion on symetric positive definite matrix
def assert_in_SPD(M):
    m,n = M.shape
    assert m == n, "matrix not square"
    assert np.allclose(M, M.T, atol=variables.absolute_numerical_threshold),"matrix not symetric"
    spectrum = np.linalg.svd(M)[1]
    largest_sv = spectrum[0]
    lowest_sv = spectrum[-1]
    assert variables.relative_numerical_threshold <= lowest_sv/largest_sv, "bad eig_1/eig_n of matrix, ie badly conditioned matrix"
    assert lowest_sv >= variables.absolute_numerical_threshold,"lowest singular value too close to zero/matrix not positive definite"

#assertion on injective application
def assert_injective_matrix(M):
    (m,n) = M.shape
    assert m>=n, "matrix not injective, issue with dimensions"
    spectrum = np.linalg.svd(M)[1]
    largest_sv = spectrum[0]
    lowest_sv = spectrum[-1]
    assert variables.relative_numerical_threshold <= lowest_sv/largest_sv, "bad eig_1/eig_n of matrix, ie badly conditioned matrix"
    assert lowest_sv>=variables.absolute_numerical_threshold,"lowest singular value too close to zero/matrix not injective"