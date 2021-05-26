import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class Matern:
    def __init__(self, degree):
        self.degree = degree
        
    def compute(self, hyp, X, X_star = None, compute_grad=False):
       N, D = X.shape
       cov_N = D + 1
       hyp_N = hyp.size

       assert(hyp_N == cov_N)
       assert(hyp.ndim == 1)
       
       ell = np.exp(hyp[0:D])
       sf2 = np.exp(2*hyp[D])
       
       f = df = tmp = None
       if self.degree == 1:
           f = lambda t : 1
           df = lambda t : 1 / t
       elif self.degree == 3:
           f = lambda t : 1 + t
           df = lambda t : 1
       elif self.degree == 5:
           f = lambda t : 1 + t * (1+ t / 3)
           df = lambda t : (1+t)/3
       else:
           assert(False)

       if X_star is None:
           # tmp = squareform(pdist(np.reshape(np.diag(np.sqrt(self.degree) / ell) @ X.T, (-1, 1))))
           tmp = np.sqrt(sq_dist(np.reshape(np.diag(np.sqrt(self.degree) / ell) @ X.T, (1, -1))))
       elif isinstance(X_star, str):
           tmp = np.zeros((X.shape[0], 1))
       else:
           a = np.reshape(np.diag(np.sqrt(self.degree) / ell) @ X.T, (1, -1))
           b = np.reshape(np.diag(np.sqrt(self.degree) / ell) @ X_star.T, (1, -1))
           #tmp = squareform(cdist(a, b))
           tmp = np.sqrt(sq_dist(a, b))
           
       # TODO: is the first multiplication right in NumPy?
       K = sf2 * f(tmp) * np.exp(-tmp)
       
       if compute_grad:
           assert(False)
        
       return K

# Might not be necessary?      
def sq_dist(a, b=None):
    D, n = a.shape
    d = m = None

    # Computation of a^2 - 2*a*b + b^2 is less stable than (a-b)^2 because numerical precision 
    # can be lost when both a and b have very large absolute value and the same sign.
    # For that reason, we subtract the mean from the data beforehand to stabilise the computations.
    # This is OK because the squared error is independent of the mean.
    if b is None:
        mu = np.mean(a, 1)
        a = a - mu
        b = a
        m = n
    else:
        d, m = b.shape
        assert(d == D)
        
        mu = (m/(n+m))*np.mean(b, 1) + (n/(n+m))*np.mean(a, 1)
        a = a - mu
        b = b - mu
    
    # Compute squared distances.
    # TODO: annoying reshapes
    C = np.reshape(np.sum(a * a, 0), (1, -1)).T + np.reshape(np.sum(b * b, 0), (1, -1)) - 2 * a.T @ b
 
    # Numerical noise can cause C to go negative, i.e. C > -1e-14
    return np.maximum(C, 0)
