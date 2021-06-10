import math
import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 

from gpyreg.f_min_fill import f_min_fill
from gpyreg.slice_sample import SliceSampler

class GP:
    def __init__(self, D, covariance, mean, noise, s2 = None):
        self.D = D
        self.covariance = covariance
        self.mean = mean
        self.noise = noise
        self.s2 = s2
        self.X = None
        self.y = None
               
    def set_priors(self, priors):
        cov_N = self.covariance.hyperparameter_count(self.D) 
        mean_N = self.mean.hyperparameter_count(self.D) 
        noise_N = self.noise.hyperparameter_count()
        hyp_N = cov_N + mean_N + noise_N
        self.hprior = HyperPrior(np.full((hyp_N,), np.nan), 
                                 np.full((hyp_N,), np.nan),
                                 np.full((hyp_N,), np.nan),
                                 np.full((hyp_N,), np.nan),
                                 np.full((hyp_N,), np.nan),
                                 np.full((hyp_N,), np.nan),
                                 np.full((hyp_N,), np.nan))

        for prior in priors:
            idx = None
            if prior == 'covariance_log_outputscale':
                idx = self.D
            elif prior == 'covariance_log_lengthscale':
                idx = range(0, self.D)
            elif prior == 'noise_log_scale':
                idx = cov_N
            elif prior == 'mean_const' :
                idx = cov_N + noise_N
            else:
                continue
                
            self.__set_priors_helper(idx, priors[prior][0], priors[prior][1])

    def __set_priors_helper(self, i, prior_type, prior_params):
        if prior_type == 'gaussian':
            mu, sigma = prior_params
            self.hprior.mu[i] = mu
            self.hprior.sigma[i] = sigma
            self.hprior.df[i] = 0
        elif prior_type == 'student_t':
            mu, sigma, df = prior_params
            self.hprior.mu[i] = mu
            self.hprior.sigma[i] = sigma
            self.hprior.df[i] = df
        elif prior_type == 'smoothbox':
            a, b, sigma = prior_params
            self.hprior.a[i] = a
            self.hprior.b[i] = b
            self.hprior.sigma[i] = sigma
        elif prior_type == 'smoothbox_student_t':
            a, b, sigma, df = prior_params
            self.hprior.a[i] = a    
            self.hprior.b[i] = b
            self.hprior.sigma[i] = sigma
            self.hprior.df[i] = df
                
    def update(self, X_new = None, y_new = None, hyp=None, compute_posterior=True):
        if X_new is not None:
            if self.X is None:
                self.X = X_new
            else:
                self.X = np.concatenate((self.X, X_new))
        
        if y_new is not None:
            if self.y is None:
                self.y = y_new
            else:
                self.y = np.concatenate((self.y, y_new))

        if hyp is not None and compute_posterior:
            hyp_N, s_N = hyp.shape
            self.post = np.empty((s_N,), dtype=Posterior)
            for i in range(0, s_N):
                self.post[i] = self.__core_computation(hyp[:, i], 0, 0)
                
    def fit(self, X = None, y = None, options={}):
        ## Default options
        opts_N = 3 # Hyperparameter optimization runs
        init_N = 2**10 # Initial design size for hyperparameter optimization
        thin = 5
        burn_in = None
        df_base = 7 # Default degrees of freedom for Student's t prior  
        s_N = options['n_samples']
        
                
        # Initialize GP if requested.
        if X is not None:
            self.X = X
        else:
            X = self.X
        if y is not None:
            self.y = y
        else:
            y = self.y
        
        cov_N = self.covariance.hyperparameter_count(self.D) 
        mean_N = self.mean.hyperparameter_count(self.D) 
        noise_N = self.noise.hyperparameter_count()
        hyp_N = cov_N + mean_N + noise_N
        hyp0 = np.zeros((hyp_N,))
        
        LB = self.hprior.LB
        UB = self.hprior.UB
        
        ## Initialize inference of GP hyperparameters (bounds, priors, etc.)
        
        cov_info = self.covariance.get_info(X, y)
        mean_info = self.mean.get_info(X, y)
        noise_info = self.noise.get_info(X, y)
        
        self.hprior.df[np.isnan(self.hprior.df)] = df_base
        
        # Set covariance/noise/mean function hyperparameter lower bounds.
        LB_cov = cov_info.LB[np.isnan(LB[0:cov_N])]
        LB_noise = noise_info.LB[np.isnan(LB[cov_N:cov_N+noise_N])]
        LB_mean = mean_info.LB[np.isnan(LB[cov_N+noise_N:cov_N+noise_N+mean_N])]
        
        # Set covariance/noise/mean function hyperparameter upper bounds.
        UB_cov = cov_info.UB[np.isnan(UB[0:cov_N])]
        UB_noise = noise_info.UB[np.isnan(UB[cov_N:cov_N+noise_N])]
        UB_mean = mean_info.UB[np.isnan(UB[cov_N+noise_N:cov_N+noise_N+mean_N])]
        
        # Create lower and upper bounds
        LB = np.concatenate([LB_cov, LB_noise, LB_mean])
        UB = np.concatenate([UB_cov, UB_noise, UB_mean])
        UB = np.maximum(LB, UB)
        
        # Plausible bounds for generation of starting points
        PLB = np.concatenate([cov_info.PLB, noise_info.PLB, mean_info.PLB])
        PUB = np.concatenate([cov_info.PUB, noise_info.PUB, mean_info.PUB])
        PLB = np.minimum(np.maximum(PLB, LB), UB)
        PUB = np.maximum(np.minimum(PUB, UB), LB)
        
        ## Hyperparameter optimization
        objective_f_1 = lambda hyp_ : self.__gp_obj_fun(hyp_, False, False)
        
        # First evaluate GP log posterior on an informed space-filling design.
        t1_s = time.time()
        X0, y0 = f_min_fill(objective_f_1, hyp0, LB, UB, PLB, PUB, self.hprior)
        hyp = X0[0:opts_N, :].T
        widths_default = np.std(X0, axis=0, ddof=1)
        
        # Extract a good low-noise starting point for the 2nd optimization.
        if noise_N > 0 and opts_N > 1 and init_N > opts_N:
            xx = X0[opts_N:, :]
            noise_y = y0[opts_N:]
            noise_params = xx[:, cov_N]
        
            # Order by noise parameter magnitude.
            order = np.argsort(noise_params)
            xx = xx[order, :]
            noise_y = noise_y[order]
            # Take the best amongst bottom 20% vectors. 
            idx_best = np.argmin(noise_y[0:math.ceil(0.2*np.size(noise_y))])
            hyp[:, 1] = xx[idx_best, :]
            
        # Fix zero widths.
        idx0 = widths_default == 0
        if np.any(idx0):
            if np.shape(hyp)[1] > 1:
                std_hyp = np.std(hyp, axis=1, ddof=1)
                widths_default[idx0] = std_hyp[idx0]
                idx0 = widths_default == 0
                
            if np.any(idx0):
                widths_default[idx0] = np.minimum(1, UB[idx0] - LB[idx0])
                
        t1 = time.time() - t1_s
        
        # Check that hyperparameters are within bounds.
        eps_LB = np.reshape(LB, (-1, 1)) + np.spacing(np.reshape(LB, (-1, 1)))
        eps_UB = np.reshape(UB, (-1, 1)) - np.spacing(np.reshape(UB, (-1, 1)))
        hyp = np.minimum(eps_UB, np.maximum(eps_LB, hyp))

        # Perform optimization from most promising NOPTS hyperparameter vectors.
        gradient = lambda hyp_ : self.__gp_obj_fun(hyp_, True, False)[1]
        objective_f_2 = lambda hyp_ : self.__gp_obj_fun(hyp_, True, False)
        nll = np.full((opts_N,), np.inf)

        # for i in range(0, 1024):
        #    res = sp.optimize.check_grad(objective_f_1, gradient, x0=X0.T[:, i])
        t2_s = time.time()
        for i in range(0, opts_N):
            # res = sp.optimize.minimize(fun=objective_f_1, x0=hyp[:, i], bounds=list(zip(LB, UB)))
            res = sp.optimize.minimize(fun=objective_f_2, x0=hyp[:, i], jac=True, bounds=list(zip(LB, UB)))
            hyp[:, i] = res.x
            nll[i] = res.fun
        
        # Take the best hyperparameter vector.
        hyp_start = hyp[:, np.argmin(nll)]
        t2 = time.time() - t2_s

        ## Sample from best hyperparameter vector using slice sampling
        
        t3_s = time.time()
        # Effective number of samples (thin after)
        eff_s_N = s_N * thin

        sample_f = lambda hyp_ : self.__gp_obj_fun(hyp_, False, True)
        #hyp_start = np.array([-2.6, 1.6, -6.9, -2.0, 2.2, 3.3])
        #widths_default = np.array([2.6, 2.6, 1.6, 4, 3.2, 2.6])
        #LB = np.array([-12, -11, -14, -20, -10, -12])
        #UB = np.array([5, 6, 4, 22, 10, 6])
        
        #hyp_start = np.array([-0.6, -0.0, -0.3, -6.9, -0.1])
        #widths_default = np.array([2.7, 2.7, 2.7, 1.5, 0.7])
        #LB = np.array([-13, -13, -14, -14, -3])
        #UB = np.array([5, 5, 4, 1, 3])
        
        options = {
            'display' : 'off',
            'diagnostics' : False
        }
        slicer = SliceSampler(sample_f, hyp_start, widths_default, LB, UB, options)
        res = slicer.sample(eff_s_N, burn=50)

        # Thin samples
        hyp_pre_thin = res.samples.T
        hyp = hyp_pre_thin[:, thin-1::thin]

        t3 = time.time() - t3_s
        print(hyp)
        print(t1, t2, t3)
        
        # Recompute GP with finalized hyperparameters.
        self.update(hyp=hyp)
        
    def __compute_log_priors(self, hyp, compute_grad):
        hyp_N = np.size(hyp)
        
        lp = 0
        dlp = None 
        if compute_grad:
            dlp = np.zeros(hyp.shape)
        
        mu = self.hprior.mu
        sigma = np.abs(self.hprior.sigma)
        df = self.hprior.df

        u_idx = (~np.isfinite(mu)) | (~np.isfinite(sigma))
        g_idx = ~u_idx & (df == 0 | ~np.isfinite(df)) & np.isfinite(sigma)
        t_idx = ~u_idx & (df > 0) & np.isfinite(df)
        
        # Quadratic form
        z2 = np.zeros(hyp.shape)
        z2[g_idx | t_idx] = ((hyp[g_idx | t_idx] - mu[g_idx | t_idx]) / sigma[g_idx | t_idx])**2
        
        # Gaussian prior
        if np.any(g_idx):
            lp -= 0.5 * (np.sum(np.log(2*np.pi*sigma[g_idx]**2)) + z2[g_idx])
            if compute_grad:
                dlp[g_idx] = -(hyp[g_idx] - mu[g_idx]) / sigma[g_idx]**2
         
        # Student's t prior
        if np.any(t_idx):
            lp += np.sum(sp.special.gammaln(0.5*(df[t_idx]+1)) - sp.special.gammaln(0.5*df[t_idx]))
            lp += np.sum(-0.5*np.log(np.pi*df[t_idx]) - np.log(sigma[t_idx]) - 0.5*(df[t_idx]+1) * np.log1p(z2[t_idx] / df[t_idx]))
            if compute_grad:
                dlp[t_idx] = -(df[t_idx]+1) / df[t_idx] / (1+z2[t_idx] / df[t_idx]) * (hyp[t_idx] - mu[t_idx]) / sigma[t_idx]**2
     
        if compute_grad:
            return lp, dlp
        else:
            return lp
        
    def __compute_nlZ(self, hyp, compute_grad, compute_prior):
        nlZ = dnlZ = None
        if compute_grad:
            nlZ, dnlZ = self.__core_computation(hyp, 1, compute_grad)
        else:
            nlZ = self.__core_computation(hyp, 1, compute_grad)
                    
        if compute_prior:
            if compute_grad:
                P, dP = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P
                dnlZ -= dP
            else:
                P = self.__compute_log_priors(hyp, compute_grad)
                nlZ -= P
                
        if compute_grad:
            return nlZ, dnlZ
        else:
            return nlZ
        
    def __gp_obj_fun(self, hyp, compute_grad, swap_sign):
        nlZ = dnlZ = None
        if compute_grad:
            nlZ, dnlZ = self.__compute_nlZ(hyp, compute_grad, self.hprior is not None)
        else:
            nlZ = self.__compute_nlZ(hyp, compute_grad, self.hprior is not None) 

        # Swap sign of negative log marginal likelihood (e.g. for sampling)
        if swap_sign:
            nlZ *= -1
            if compute_grad:
                dnlZ *= -1
        
        if compute_grad:
            return nlZ, dnlZ
        else:
            return nlZ
        
    def predict(self, x_star, y_star = None, s2_star = 0, add_noise=False):
        if x_star.ndim == 1:
            x_star = np.reshape(x_star, (-1, 1))
        N, D = self.X.shape
        s_N = self.post.size
        N_star = x_star.shape[0]
        
        # Preallocate space
        fmu = np.zeros((N_star, s_N))
        ymu = np.zeros((N_star, s_N))
        fs2 = np.zeros((N_star, s_N))
        ys2 = np.zeros((N_star, s_N))
        lp = np.array([])
        
        for s in range(0, s_N):
            hyp = self.post[s].hyp
            alpha = self.post[s].alpha
            L = self.post[s].L
            L_chol = self.post[s].L_chol
            sW = self.post[s].sW
            sn2_mult = self.post[s].sn2_mult
            
            cov_N = self.covariance.hyperparameter_count(D)
            mean_N = self.mean.hyperparameter_count(D)
            noise_N = self.noise.hyperparameter_count()
            sn2_star = self.noise.compute(hyp[cov_N:cov_N+noise_N], x_star, y_star, s2_star)
            m_star = np.reshape(self.mean.compute(hyp[cov_N+noise_N:cov_N+noise_N+mean_N], x_star), (-1, 1))
            Ks = self.covariance.compute(hyp[0:cov_N], self.X, x_star)
            kss = self.covariance.compute(hyp[0:cov_N], x_star, "diag")

            if N > 0:
                fmu[:, s:s+1] = m_star + np.dot(Ks.T, alpha) # Conditional mean
            else:
                fmu[:, s:s+1] = m_star
                
            ymu[:, s] = fmu[:, s]
            if N > 0:
                if L_chol:
                    V = np.linalg.solve(L.T, np.tile(sW, (1, N_star)) * Ks)
                    fs2[:, s:s+1] = kss - np.reshape(np.sum(V * V, 0), (-1, 1)) # predictive variance
                else:
                    fs2[:, s:s+1] = kss + np.reshape(np.sum(Ks * np.dot(L, Ks), 0), (-1, 1))
            else:
                fs2[:, s:s+1] = kss
                
            fs2[:, s] = np.maximum(fs2[:, s], 0) # remove numerical noise, i.e. negative variances
            ys2[:, s:s+1] = fs2[:, s:s+1] + sn2_star * sn2_mult

        # Unless predictions for samples are requested separately, average over samples.
        if s_N > 1:
            fbar = np.reshape(np.sum(fmu, 1), (-1, 1))/ s_N
            ybar = np.reshape(np.sum(ymu, 1), (-1, 1)) / s_N
            vf = np.sum((fmu-fbar)**2, 1) / (s_N - 1)
            fs2 = np.reshape(np.sum(fs2, 1) / s_N + vf, (-1, 1))
            vy = np.sum((ymu-ybar)**2, 1) / (s_N - 1)
            ys2 = np.reshape(np.sum(ys2, 1) / s_N + vy, (-1, 1))
        
            fmu = fbar
            ymu = ybar
        
        if add_noise:
            return ymu, ys2
        return fmu, fs2

    # sigma doesn't work, requires gplite_quad implementation
    # quantile doesn't work, requires gplite_qpred implementation
    def plot(self, x0 = None, lb = None, ub = None, max_min_flag = None):
        delta_y = None
        if np.isscalar(lb) and ub is None:
            delta_y = lb
            lb = None
            
        N, D = self.X.shape # Number of training points and dimension
        s_N = self.post.size # Hyperparameter samples
        x_N = 100 # Grid points per visualization
        
        # Loop over hyperparameter samples.
        ell = np.zeros((D, s_N))
        for s in range(0, s_N):
            ell[:, s] = np.exp(self.post[s].hyp[0:D]) # Extract length scale from HYP
        ellbar = np.sqrt(np.mean(ell**2, 1)).T

        if lb is None:
            lb = np.min(self.X, axis=0) - ellbar
        if ub is None:
            ub = np.max(self.X, axis=0) + ellbar
            
        gutter = [0.05, 0.05]
        margins = [0.1, 0.01, 0.12, 0.01]
        linewidth = 1

        if x0 is None:
            max_min_flag = True
        if max_min_flag is not None:
            if max_min_flag:
                i = np.argmax(self.y)
                x0 = self.X[i, :]
            else:
                i = np.argmin(self.y)
                x0 = self.X[i, :]

        fig, ax = plt.subplots(D, D, squeeze=False)
     
        flo = fhi = None
        for i in range(0, D):
            ax[i, i].set_position(self.__tight_subplot(D, D, i, i, gutter, margins))
            
            xx = None
            xx_vec = np.reshape(np.linspace(lb[i], ub[i], np.ceil(x_N**1.5).astype(int)), (-1, 1))
            if D > 1:
                xx = np.tile(x0, (np.size(xx_vec), 1))
                xx[:, i:i+1] = xx_vec
            else:
                xx = xx_vec
            
            # TODO: Missing quantile prediction stuff etc here
            fmu, fs2 = self.predict(xx, add_noise=False)
            flo = fmu - 1.96 * np.sqrt(fs2)
            fhi = fmu + 1.96 * np.sqrt(fs2)
               
            if delta_y is not None:
                # Probably doesn't work
                fm0, _ = self.predict(x0, add_noise=False)
                dx = xx_vec[1] - xx_vec[0]
                region = np.abs(fmu-fmu0) < delta_y
                if np.any(region):
                    idx1 = np.argmax(region)
                    idx2 = np.size(region) - np.argmax(region[::-1]) - 1
                    LB[i] = x0[idx1] - 0.5 * dx
                    UB[i] = x0[idx2] + 0.5 * dx
                else:
                    LB[i] = x0[i] - 0.5 * dx
                    UB[i] = x0[i] + 0.5 * dx
                    
                xx_vec = np.reshape(np.linspace(lb[i], ub[i], np.ceil(x_N**1.5).astype(int)), (-1, 1))
                if D > 1:
                    xx = np.tile(x0, (np.size(xx_vec), 1))
                    xx[:, i:i+1] = xx_vec
                else:
                    xx = xx_vec
                    
                # TODO: Missing quantile prediction stuff etc here
                fmu, fs2 = self.predict(xx, add_noise=False)
                flo = fmu - 1.96 * np.sqrt(fs2)
                fhi = fmu + 1.96 * np.sqrt(fs2)
            
            ax[i, i].plot(xx_vec, fmu, '-k', linewidth=linewidth)
            ax[i, i].plot(xx_vec, fhi, '-', color=(0.8, 0.8, 0.8), linewidth=linewidth)
            ax[i, i].plot(xx_vec, flo, '-', color=(0.8, 0.8, 0.8), linewidth=linewidth)
            ax[i, i].set_xlim(lb[i], ub[i])
            ax[i, i].set_ylim(ax[i, i].get_ylim())
            
            # ax[i, i].tick_params(direction='out')
            ax[i, i].spines["top"].set_visible(False)
            ax[i, i].spines["right"].set_visible(False)
            
            if D == 1:
                ax[i, i].set_xlabel('x')
                ax[i, i].set_ylabel('y')
                ax[i, i].scatter(self.X, self.y, color="blue")
            else:
                if i == 0:
                    ax[i, i].set_ylabel(r"$x_" + str(i+1) + r"$")
                if i == D - 1:
                    ax[i, i].set_xlabel(r"$x_" + str(i+1) + r"$")
            ax[i, i].vlines(x0[i], ax[i, i].get_ylim()[0], ax[i, i].get_ylim()[1], colors='k', linewidth=linewidth)

        for i in range(0, D):
            for j in range(0, i):
                xx1_vec = np.reshape(np.linspace(lb[i], ub[i], x_N), (-1, 1)).T
                xx2_vec = np.reshape(np.linspace(lb[j], ub[j], x_N), (-1, 1)).T
                xx_vec = np.array(np.meshgrid(xx1_vec, xx2_vec)).T.reshape(-1, 2)
                
                xx = np.tile(x0, (x_N*x_N, 1))
                xx[:, i] = xx_vec[:, 0]
                xx[:, j] = xx_vec[:, 1]
                 
                fmu, fs2 = self.predict(xx, add_noise=False)
                
                for k in range(0, 2):
                    i1 = i2 = mat = None
                    if k == 1:
                        i1 = j
                        i2 = i
                        mat = np.reshape(fmu, (x_N, x_N)).T
                    else:
                        i1 = 1
                        i2 = j
                        mat = np.reshape(np.sqrt(fs2), (x_N, x_N))    
                    ax[i1, i2].set_position(self.__tight_subplot(D, D, i1, i2, gutter, margins))
                    ax[i1, i2].spines["top"].set_visible(False)
                    ax[i1, i2].spines["right"].set_visible(False)
                    
                    if k == 1:
                        Xt, Yt = np.meshgrid(xx1_vec, xx2_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    else:
                        Xt, Yt = np.meshgrid(xx2_vec, xx1_vec)
                        ax[i1, i2].contour(Xt, Yt, mat)
                    ax[i1, i2].set_xlim(lb[i2], ub[i2])
                    ax[i1, i2].set_ylim(lb[i1], ub[i1])
                    ax[i1, i2].scatter(self.X[:, i2], self.X[:, i1], color="blue", s=10)
                    
                    ax[i1, i2].hlines(x0[i1], ax[i1, i2].get_xlim()[0], ax[i1, i2].get_xlim()[1], colors='k', linewidth=linewidth)
                    ax[i1, i2].vlines(x0[i2], ax[i1, i2].get_ylim()[0], ax[i1, i2].get_ylim()[1], colors='k', linewidth=linewidth)
                    
                if j == 0:
                    ax[i, j].set_ylabel(r"$x_" + str(i+1) + r"$")
                if i == D-1:
                    ax[i, j].set_xlabel(r"$x_" + str(j+1) + r"$")

        plt.show()
        
    def __tight_subplot(self, m, n, row, col, gutter = [.002, .002], margins = [.06, .01, .04, .04]):
        Lmargin = margins[0]
        Rmargin = margins[1]
        Bmargin = margins[2]
        Tmargin = margins[3]
        
        unit_height = (1 - Bmargin - Tmargin - (m-1)*gutter[1]) / m
        height = np.size(row) * unit_height + (np.size(row) - 1) * gutter[1]
        
        unit_width = (1 - Lmargin - Rmargin - (n-1)*gutter[0]) / n
        width = np.size(col) * unit_width + (np.size(col) - 1) * gutter[0]
        
        bottom = (m - np.max(row) - 1) * (unit_height + gutter[1]) + Bmargin
        left = np.min(col) * (unit_width + gutter[0]) + Lmargin

        pos_vec = [left, bottom, width, height]
        
        return pos_vec
        
    def __core_computation(self, hyp, compute_nlZ, compute_nlZ_grad):
        N, d = self.X.shape
        cov_N = self.covariance.hyperparameter_count(d)
        mean_N = self.mean.hyperparameter_count(d)
        noise_N = self.noise.hyperparameter_count()
        sn2 = m = K = dsn2 = dm = dK = None
        if compute_nlZ_grad:
            sn2, dsn2 = self.noise.compute(hyp[cov_N:cov_N+noise_N], self.X, self.y, self.s2, compute_grad=True)
            m, dm = self.mean.compute(hyp[cov_N+noise_N:cov_N+noise_N+mean_N], self.X, compute_grad=True)
            m = np.reshape(m, (-1, 1))
            K, dK = self.covariance.compute(hyp[0:cov_N], self.X, compute_grad=True)   
        else:
            sn2 = self.noise.compute(hyp[cov_N:cov_N+noise_N], self.X, self.y, self.s2)
            m = np.reshape(self.mean.compute(hyp[cov_N+noise_N:cov_N+noise_N+mean_N], self.X), (-1, 1))
            K = self.covariance.compute(hyp[0:cov_N], self.X)   
        sn2_mult = 1 # Effective noise variance multiplier 
        
        L_chol = np.min(sn2) >= 1e-6
        L = sl = pL = None
        if L_chol:
            sn2_div = sn2_mat = None
            if np.isscalar(sn2):
                sn2_div = sn2
                sn2_mat = np.eye(N)
            else:
                sn2_div = np.min(sn2)
                sn2_mat = np.diag(sn2 / sn2_div) 
                 
            for i in range(0, 10):
                try:
                    L = sp.linalg.cholesky(K / (sn2_div * sn2_mult) + sn2_mat)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = sn2_div * sn2_mult
            pL = L
        else:
            if np.isscalar(sn2):
                sn2_mat = sn2 * np.eye(N)
            else:
                sn2_mat = np.diag(sn2)
            for i in range(0, 10):
                try:
                    L = sp.linalg.cholesky(K + sn2_mult * sn2_mat)
                except sp.linalg.LinAlgError:
                    sn2_mult *= 10
                    continue
                break
            sl = 1
            pL = np.linalg.solve(-L, np.linalg.solve(L.T, np.eye(N)))
 
        alpha = np.linalg.solve(L, np.linalg.solve(L.T, self.y - m)) / sl
        
        # Negative log marginal likelihood computation
        if compute_nlZ:
            hyp_N = np.size(hyp)
            nlZ = np.dot((self.y - m).T, alpha/2) + np.sum(np.log(np.diag(L))) + N * np.log(2*np.pi*sl)/2
            
            if compute_nlZ_grad:
                dnlZ = np.zeros(hyp.shape)
                Q = np.linalg.solve(L, np.linalg.solve(L.T, np.eye(N))) / sl - np.dot(alpha, alpha.T)
         
                # Gradient of covariance hyperparameters.
                for i in range(0, cov_N):
                    dnlZ[i] = np.sum(np.sum(Q * dK[:, :, i])) / 2
                    
                # Gradient of GP likelihood
                if np.isscalar(sn2):
                    tr_Q = np.trace(Q)
                    for i in range(0, noise_N):
                        dnlZ[cov_N+i] = 0.5 * sn2_mult * np.dot(dsn2[i], tr_Q)
                else:
                    dg_Q = np.diag(Q)
                    for i in range(0, noise_N):
                        dnlZ[cov_N+i] = 0.5 * sn2_mult * np.sum(dsn2[:, i] * dg_Q)
               
                # Gradient of mean function.
                if mean_N > 0:
                    dnlZ[cov_N + noise_N:] = np.dot(-dm.T, alpha)[:, 0]
                    
                return nlZ[0, 0], dnlZ
                
            return nlZ[0, 0]
     
        return Posterior(hyp, alpha, np.ones((N, 1)) / np.sqrt(np.min(sn2)*sn2_mult), pL, sn2_mult, L_chol)
        
class HyperPrior:
    def __init__(self, mu, sigma, df, a, b, LB, UB):
        self.mu = mu
        self.sigma = sigma
        self.df = df
        self.a = a
        self.b = b
        self.LB = LB
        self.UB = UB
                
class Posterior:
    def __init__(self, hyp, alpha, sW, L, sn2_mult, Lchol):
        self.hyp = hyp
        self.alpha = alpha
        self.sW = sW
        self.L = L
        self.sn2_mult = sn2_mult
        self.L_chol = Lchol
