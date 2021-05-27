import numpy as np

class GaussianNoise:
    def __init__(self, parameters):
        self.parameters = parameters
        
    def hyperparameter_count(self):
        noise_N = 0
        if self.parameters[0] == 1:
           noise_N += 1
        if self.parameters[1] == 2:
           noise_N += 1
        if self.parameters[2] == 1:
           noise_N += 2
        return noise_N
        
    def compute(self, hyp, X, y, s2, compute_grad=False):
        N, D = X.shape
        hyp_N = hyp.size
        
        # Compute number of likelihood function hyperparameters.
        noise_N = self.hyperparameter_count()
           
        assert(hyp_N == noise_N)
        assert(hyp.ndim == 1)
        
        if compute_grad:
            assert(False)
            
        i = 0
        sn2 = 0
        if self.parameters[0] == 0:
            sn2 = np.spacing(1.0)
        else:
            sn2 = np.exp(2*hyp[i])
            i += 1
            
        if self.parameters[1] == 1:
            sn2 += s2
        elif self.parameters[1] == 2:
            sn2 += np.exp(hyp[i]) * s2
            i += 1
            
        if self.parameters[2] == 1:
            if y is not None:
                y_tresh = hyp[i]
                w2 = np.ewxp(2*hyp[i+1])
                zz = np.maximum(0, y_tresh - y)
                
                sn2 += w2 @ zz ** 2
            i += 2
        
        return sn2   
