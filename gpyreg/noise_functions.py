import numpy as np

class GaussianNoise:
    def __init__(self, constant_add = False, user_provided_add = False, scale_user_provided = False, rectified_linear_output_dependent_add = False):
        self.parameters = np.zeros((3,))
        if constant_add:
            self.parameters[0] = 1
        if user_provided_add: 
            self.parameters[1] = 1
            if scale_user_provided:
                self.parameters[1] += 1
        if rectified_linear_output_dependent_add:
            self.parameters[2] = 1 
        
    def hyperparameter_count(self):
        noise_N = 0
        if self.parameters[0] == 1:
           noise_N += 1
        if self.parameters[1] == 2:
           noise_N += 1
        if self.parameters[2] == 1:
           noise_N += 2
        return noise_N
        
    def hyperparameter_info(self):
        hparams = []
        if self.parameters[0] == 1:
            hparams.append(('noise_log_scale', 1))
        if self.parameters[1] == 2:
            hparams.append(('noise_placeholder_hyperparams_1', 1))
        if self.parameters[2] == 1:
            hparams.append(('noise_placeholder_hyperparams_2', 2))
            
        return hparams
        
    def get_info(self, X, y):
        N, D = X.shape
        noise_N = self.hyperparameter_count()
        tol = 1e-6
        dsn2 = NoiseInfo(np.full((noise_N,), -np.inf),
                         np.full((noise_N,), np.inf),
                         np.full((noise_N,), -np.inf),
                         np.full((noise_N,), np.inf),
                         np.full((noise_N,), np.nan))
        
        if np.size(y) <= 1:
            y = np.array([0, 1])
        height = np.max(y) - np.min(y)    

        i = 0
        # Base constant noise
        if self.parameters[0] == 1:
            # Constant noise (log standard deviation)
            dsn2.LB[i] = np.log(tol)
            dsn2.UB[i] = np.log(height)
            dsn2.PLB[i] = 0.5 * np.log(tol)
            dsn2.PUB[i] = np.log(np.std(y, ddof=1))
            dsn2.x0[i] = np.log(1e-3)
            i += 1
            
        # User provided noise.
        if self.parameters[1] == 2:
            dsn2.LB[i] = np.log(1e-3)
            dsn2.UB[i] = np.log(1e3)
            dsn2.PLB[i] = np.log(0.5)
            dsn2.PUB[i] = np.log(2)
            dsn2.x0[i] = np.log(1)
            i += 1
            
        # Output dependent noise
        if self.parameters[2] == 1:
            min_y = np.min(y)
            max_y = np.max(y)
            dsn2.LB[i] = min_y
            dsn2.UB[i] = max_y
            dsn2.PLB[i] = min_y
            dsn2.PUB[i] = np.max(max_y - 5 * D, min_y) 
            dsn2.x0[i] = np.max(max_y - 10 * D, min_y)
            i += 1
            
            dsn2.LB[i] = np.log(1e-3)
            dsn2.UB[i] = np.log(0.1)
            dsn2.PLB[i] = np.log(0.01)
            dsn2.PUB[i] = np.log(0.1)
            dsn2.x0[i] = np.log(0.1)
            i += 1
            
        # Plausible starting point
        i_nan = np.isnan(dsn2.x0)
        dsn2.x0[i_nan] = 0.5 * (dsn2.PLB[i_nan] + dsn2.PUB[i_nan])
        
        return dsn2
        
    def compute(self, hyp, X, y, s2, compute_grad=False):
        N, D = X.shape
        noise_N = self.hyperparameter_count()
           
        if hyp.size != noise_N:
            raise ValueError('Expected %d noise function hyperparameters, %d passed instead.' % (noise_N, hyp_N))
        if hyp.ndim != 1:
            raise ValueError('Noise function output is available only for one-sample hyperparameter inputs.')
        
        dsn2 = None
        if compute_grad:
            if any(x > 0 for x in self.parameters[1:]):
                dsn2 = np.zeros((N, noise_N))
            else:
                dsn2 = np.zeros((1, noise_N))
            
        i = 0
        sn2 = 0
        if self.parameters[0] == 0:
            sn2 = np.spacing(1.0)
        else:
            sn2 = np.exp(2*hyp[i])
            if compute_grad:
                dsn2[:, i] = 2 * sn2
            i += 1
            
        if self.parameters[1] == 1:
            sn2 += s2
        elif self.parameters[1] == 2:
            sn2 += np.exp(hyp[i]) * s2
            if compute_grad:
                dsn2[:, i] = np.exp(hyp[i]) * s2
            i += 1
            
        if self.parameters[2] == 1:
            if y is not None:
                y_tresh = hyp[i]
                w2 = np.exp(2*hyp[i+1])
                zz = np.maximum(0, y_tresh - y)
                
                sn2 += w2 @ zz ** 2
                if compute_grad:
                    dsn2[:, i] = 2 *  w2 * (y_tresh - y) * (zz > 0)
                    dsn2[:, i+1] = 2 * w2 * zz**2
            i += 2
        
        if compute_grad:
            return sn2, dsn2
        
        return sn2   
        
class NoiseInfo:
    def __init__(self, LB, UB, PLB, PUB, x0):
        self.LB = LB
        self.UB = UB
        self.PLB = PLB
        self.PUB = PUB
        self.x0 = x0
