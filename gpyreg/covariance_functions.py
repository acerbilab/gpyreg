import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class SquaredExponential:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return d + 1
        
    def get_info(self, X, y):
        return CovarianceInfo(self, X, y)
    
    def compute(self, hyp, X, X_star = None, compute_grad = False):
       N, D = X.shape
       cov_N = self.hyperparameter_count(D)
       hyp_N = hyp.size

       assert(hyp_N == cov_N)
       assert(hyp.ndim == 1)
       
       ell = np.exp(hyp[0:D])
       sf2 = np.exp(2*hyp[D])
       
       tmp = None
       if X_star is None:
           tmp = squareform(pdist(X @ np.diag(1 / ell), 'sqeuclidean'))
       elif isinstance(X_star, str):
           tmp = np.zeros((X.shape[0], 1))
       else:
           tmp = cdist(X @ np.diag(1 / ell), X_star @ np.diag(1 / ell), 'sqeuclidean')
           
       K = sf2 * np.exp(-tmp/2)
       
       if compute_grad:
           dK = np.zeros((N, N, cov_N))
           for i in range(0, D):
               dK[:, :, i] = K * squareform(pdist(np.reshape(X[:, i] / ell[i], (1, -1)), 'sqeuclidean'))
           dK[:, :, D] = 2 * K
           return K, dK

       return K

class Matern:
    def __init__(self, degree):
        self.degree = degree
        
    def hyperparameter_count(self, d):
        return d + 1
        
    def get_info(self, X, y):
        return CovarianceInfo(self, X, y)
        
    def compute(self, hyp, X, X_star = None, compute_grad = False):
       N, D = X.shape
       cov_N = self.hyperparameter_count(D)
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
           tmp = squareform(pdist(X @ np.diag(np.sqrt(self.degree) / ell)))
       elif isinstance(X_star, str):
           tmp = np.zeros((X.shape[0], 1))
       else:
           a = X @ np.diag(np.sqrt(self.degree) / ell)
           b = X_star @ np.diag(np.sqrt(self.degree) / ell)
           tmp = cdist(a, b)
           
       K = sf2 * f(tmp) * np.exp(-tmp)

       if compute_grad:
           dK = np.zeros((N, N, cov_N))
           for i in range(0, D):
               Ki = squareform(pdist(np.reshape(np.sqrt(self.degree) / ell[i] * X[:, i], (-1, 1)), 'sqeuclidean'))
               dK[:, :, i] = sf2 * (df(tmp) * np.exp(-tmp)) * Ki
           dK[:, :, D] = 2 * K
           return K, dK

       return K
 
class CovarianceInfo:
    def __init__(self, gp, X, y):
        N, D = X.shape
        cov_N = gp.hyperparameter_count(D)
        tol = 1e-6
        self.LB = np.full((cov_N,), -np.inf)
        self.UB = np.full((cov_N,), np.inf)
        self.PLB = np.full((cov_N,), -np.inf)
        self.PUB = np.full((cov_N,), np.inf)
        self.x0 = np.full((cov_N,), np.nan)
        
        width = np.max(X, axis=0) - np.min(X, axis=0)
        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)    

        self.LB[0:D] = np.log(width) + np.log(tol)
        self.UB[0:D] = np.log(width * 10)
        self.PLB[0:D] = np.log(width) + 0.5 * np.log(tol)
        self.PUB[0:D] = np.log(width)
        self.x0[0:D] = np.log(np.std(X, ddof=1))
        
        self.LB[D] = np.log(height) + np.log(tol)
        self.UB[D] = np.log(height * 10)
        self.PLB[D] = np.log(height) + 0.5 * np.log(tol)
        self.PUB[D] = np.log(height)
        self.x0[D] = np.log(np.std(y, ddof=1))
        
        # Plausible starting point
        i_nan = np.isnan(self.x0)
        self.x0[i_nan] = 0.5 * (self.PLB[i_nan] + self.PUB[i_nan])
