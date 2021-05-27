import numpy as np

class ZeroMean:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 0
        
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        m = np.zeros((N, 1))
        
        if compute_grad:
            assert(False)
            
        return m 
        
class ConstantMean:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 1
        
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        m0 = hyp[0]
        m = m0 * np.ones((N, 1))
        
        if compute_grad:
            assert(False)
            
        return m

class NegativeQuadratic:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 1 + 2 * d
    
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)
        hyp_N = hyp.size

        assert(hyp_N == mean_N)
        assert(hyp.ndim == 1)
        
        m_0 = hyp[0]
        x_m = hyp[1:(1+D)]
        omega = np.exp(hyp[(1+D):(1+2*D)])
        z_2 = ((X - x_m) / omega)**2
        m = m_0 - 0.5 * np.sum(z_2, 1)

        if compute_grad:
            assert(False)
            
        return m
