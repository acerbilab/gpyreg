import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 

class GP:
    def __init__(self, D, covariance, mean, noise, s2 = None):
        self.D = D
        self.covariance = covariance
        self.mean = mean
        self.noise = noise
        self.s2 = s2
               
    def set_priors(self, gp_priors):
        pass
         
    def update(self, hyp, X, y, compute_posterior=True):
        self.X = X
        self.y = y
 
        if compute_posterior:
            hyp_N, s_N = hyp.shape
            self.post = np.empty((s_N,), dtype=Posterior)
            for i in range(0, s_N):
                self.post[i] = self.__core_computation(hyp[:, i], 0, 0)
        
    def fit(self, x, y, options):
        pass
        
    def predict(self, x_star, y_star = None, s2_star = None, add_noise=False):
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
            ys2[:, s] = fs2[:, s] + sn2_star * sn2_mult

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
        
        return ymu, ys2, fmu, fs2

    def plot(self, x0 = None, lb = None, ub = None, sigma = None, quantile = False):
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
            x0 = "max"
        if isinstance(x0, str):
            if x0 == "max":
                i = np.argmax(self.y)
                x0 = self.X[i, :]
            elif x0 == "min":
                i = np.argmin(self.y)
                x0 = self.X[i, :]
            else:
                assert(False)      
                
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
            
            if sigma is None:
                if quantile:
                    assert(False)
                else:
                    _, _, fmu, fs2 = self.predict(xx)
                    flo = fmu - 1.96 * np.sqrt(fs2)
                    fhi = fmu + 1.96 * np.sqrt(fs2)
            else:
                assert(False)
               
            if delta_y is not None:
                assert(False)
            
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
                 
                if sigma is not None:
                   assert(False)
                   
                _, _, fmu, fs2 = self.predict(xx)
                
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
        sn2 = self.noise.compute(hyp[cov_N:cov_N+noise_N], self.X, self.y, self.s2)
        sn2_mult = 1 # Effective noise variance multiplier 
        m = np.reshape(self.mean.compute(hyp[cov_N+noise_N:cov_N+noise_N+mean_N], self.X), (-1, 1))
        K = self.covariance.compute(hyp[0:cov_N], self.X)

        L_chol = np.min(sn2) >= 1e-6
        L = sl = pL = None
        if L_chol:
            sn2_div = sn2_mat = None
            if np.isscalar(sn2):
                sn2_div = sn2
                sn2_mat = np.eye(N)
            else:
                sn2_div = np.min(sn2)
                sn2_mat = np.diag(sn2 / sn2div) 
                 
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
            assert(False)
            
            if compute_nlZ_grad:
                assert(False)
     
        return Posterior(hyp, alpha, np.ones((N, 1)) / np.sqrt(np.min(sn2)*sn2_mult), pL, sn2_mult, L_chol)
                
class Posterior:
    def __init__(self, hyp = None, alpha = None, sW = None, L = None, sn2_mult = None, Lchol = None):
        self.hyp = hyp
        self.alpha = alpha
        self.sW = sW
        self.L = L
        self.sn2_mult = sn2_mult
        self.L_chol = Lchol
