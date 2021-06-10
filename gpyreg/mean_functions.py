import numpy as np

class ZeroMean:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 0
        
    def get_info(self, X, y):
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 0)
        
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise Exception('Expected %d mean function hyperparameters, %d passed instead.' % (noise_N, hyp_N))
        if hyp.ndim != 1:
            raise Exception('Mean function output is available only for one-sample hyperparameter inputs.')

        m = np.zeros((N, 1))
        
        if compute_grad:
            return m, []
            
        return m 
        
class ConstantMean:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 1
        
    def get_info(self, X, y):
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 1)
        
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise Exception('Expected %d mean function hyperparameters, %d passed instead.' % (noise_N, hyp_N))
        if hyp.ndim != 1:
            raise Exception('Mean function output is available only for one-sample hyperparameter inputs.')
        
        m0 = hyp[0]
        m = m0 * np.ones((N, 1))
        
        if compute_grad:
            return m, np.ones((N, 1))
            
        return m

class NegativeQuadratic:
    def __init__(self):
        pass
        
    def hyperparameter_count(self, d):
        return 1 + 2 * d
        
    def get_info(self, X, y):
        mean_N = self.hyperparameter_count(X.shape[1])
        return MeanInfo(mean_N, X, y, 2)
    
    def compute(self, hyp, X, compute_grad = False):
        N, D = X.shape
        mean_N = self.hyperparameter_count(D)

        if hyp.size != mean_N:
            raise Exception('Expected %d mean function hyperparameters, %d passed instead.' % (noise_N, hyp_N))
        if hyp.ndim != 1:
            raise Exception('Mean function output is available only for one-sample hyperparameter inputs.')
        
        m_0 = hyp[0]
        x_m = hyp[1:(1+D)]
        omega = np.exp(hyp[(1+D):(1+2*D)])
        z_2 = ((X - x_m) / omega)**2
        m = m_0 - 0.5 * np.sum(z_2, 1)

        if compute_grad:
            dm = np.zeros((N, mean_N))
            dm[:, 0] = np.ones((N,))
            dm[:, 1:D+1] = ((X - x_m) / omega**2)
            dm[:, D+1:] = z_2
            return m, dm
            
        return m
        
class MeanInfo:
    def __init__(self, mean_N, X, y, idx):
        N, D = X.shape
        tol = 1e-6
        big = np.exp(3)
        self.LB = np.full((mean_N,), -np.inf)
        self.UB = np.full((mean_N,), np.inf)
        self.PLB = np.full((mean_N,), -np.inf)
        self.PUB = np.full((mean_N,), np.inf)
        self.x0 = np.full((mean_N,), np.nan)
        
        w = np.max(X) - np.min(X)
        if np.size(y) <= 1:
            y = np.array([0, 1])
        h = np.max(y) - np.min(y)    

        if idx == 0:
            pass
        elif idx == 1:
            self.LB[0] = np.min(y) - 0.5 * h
            self.UB[0] = np.max(y) + 0.5 * h
            # For future reference note that quantile behaviour in MATLAB and NumPy is slightly different.
            # https://stackoverflow.com/questions/58424704/output-produced-by-python-numpy-percentile-not-same-as-matlab-prctile
            self.PLB[0] = np.quantile(y, 0.1)
            self.PUB[0] = np.quantile(y, 0.9)
            self.x0[0] = np.median(y)
        else:
            self.LB[0] = np.min(y)
            self.UB[0] = np.max(y) + h
            self.PLB[0] = np.median(y)
            self.PUB[0] = np.max(y)
            self.x0[0] = np.quantile(y, 0.9)   
            
            # xm
            self.LB[1:1+D] = np.min(X) - 0.5 * w
            self.UB[1:1+D] = np.max(X) + 0.5 * w
            self.PLB[1:1+D] = np.min(X)
            self.PUB[1:1+D] = np.max(X)
            self.x0[1:1+D] = np.median(X)

            # omega
            self.LB[1+D:mean_N] = np.log(w) + np.log(tol)
            self.UB[1+D:mean_N] = np.log(w) + np.log(big)
            self.PLB[1+D:mean_N] = np.log(w) + 0.5 * np.log(tol)
            self.PUB[1+D:mean_N] = np.log(w)
            # For future reference note that std behaviour in MATLAB and NumPy is slightly different.
            # https://stackoverflow.com/questions/27600207/why-does-numpy-std-give-a-different-result-to-matlab-std
            self.x0[1+D:mean_N] = np.log(np.std(X, ddof=1))     
        
        # Plausible starting point
        i_nan = np.isnan(self.x0)
        self.x0[i_nan] = 0.5 * (self.PLB[i_nan] + self.PUB[i_nan])
