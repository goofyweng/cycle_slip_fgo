import algebra as alg
import numpy as np

#whitening of system: from covariance R to identity covariance and transformed matrix and vector
def whitening(y,H,R):
    alg.assert_in_SPD(R)
    L = np.linalg.cholesky(R)
    Linv = alg.invert_matrix(L)
    Linvy = Linv@y
    LinvH = Linv@H
    return (Linvy, LinvH)

def projectors(H):
    (m,n) = H.shape
    alg.assert_injective_matrix(H)
    S = (alg.invert_matrix(H.T@H))@(H.T)
    HS = H@S
    IHS = np.eye(m)-HS
    return S,HS,IHS

def leastSquares(y,H,R):
    #y an (m,1) vector
    #H an injective (m,n) matrix
    #R a covariance matrix (in SPD(R))
    #returns estimate

    yw,Hw = whitening(y,H,R)
    S,HS,IHS = projectors(Hw)
    xhat = S@yw
    return xhat