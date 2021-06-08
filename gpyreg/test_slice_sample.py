import numpy as np
import matplotlib.pyplot as plt 

from scipy.stats import norm, expon, uniform, beta, multivariate_normal, multivariate_t

from gpyreg.slice_sample import SliceSampler

# These unit tests can fail with some small probability. 

options = {
    'display' : 'off',
    'diagnostics' : False
}
threshold = 0.1

def test_normal():
    slicer = SliceSampler(norm.logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000).samples
    
    assert np.abs(norm.mean() - np.mean(samples)) < threshold
    assert np.abs(norm.var() - np.var(samples)) < threshold
    
def test_normal_mixture():
    p = 0.7
    rv1 = norm(0, 1)
    rv2 = norm(6, 2)
    pdf = lambda x : p * rv1.pdf(x) + (1-p)*rv2.pdf(x)
    logpdf = lambda x : np.log(pdf(x)) # if pdf(x) > np.spacing(0) else -np.inf
    slicer = SliceSampler(logpdf, np.array([0.5]), options=options)
    samples = slicer.sample(20000).samples
  
    assert np.abs((1-p)*6 - np.mean(samples)) < threshold
    # plt.scatter(samples , pdf(samples))
    # plt.show()
    
def test_exponential():
    slicer = SliceSampler(expon.logpdf, np.array([0.5]), LB=0.0, options=options)
    samples = slicer.sample(20000).samples
    
    # plt.scatter(samples , expon.pdf(samples))
    # plt.show()
    assert np.abs(expon.mean() - np.mean(samples)) < threshold
    assert np.abs(expon.var() - np.var(samples)) < threshold
    
def test_uniform():
    slicer = SliceSampler(uniform.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options)
    samples = slicer.sample(20000).samples
   
    assert np.abs(uniform.mean() - np.mean(samples)) < threshold
    assert np.abs(uniform.var() - np.var(samples)) < threshold
    
def test_beta():
    a, b = 2.31, 0.627
    rv = beta(a, b)
    slicer = SliceSampler(rv.logpdf, np.array([0.5]), LB=0.0, UB=1.0, options=options)
    samples = slicer.sample(20000).samples
    
    assert np.abs(rv.mean() - np.mean(samples)) < threshold
    assert np.abs(rv.var() - np.var(samples)) < threshold
    
def test_multivariate_normal():
    mean = np.array([0.68, 0.6, 0.4])
    cov = np.array([[1.58, 0.96, -1.2], 
                    [0.96, 2.17, -1.725], 
                    [-1.2, -1.725, 1.85]])
    rv = multivariate_normal(mean, cov)
    slicer = SliceSampler(rv.logpdf, np.array([0.5, -0.5, 1.0]), options=options)
    samples = slicer.sample(20000).samples

    assert np.all(np.abs(mean - np.mean(samples, axis=0)) < threshold)
    # assert np.all(np.abs(cov - np.cov(samples.T)) < threshold)
    
def test_multivariate_t():
    x = [1.0, -0.5]
    loc = [[2.1, 0.3], [0.3, 1.5]]
    rv = multivariate_t(x, loc, df=3)
    slicer = SliceSampler(rv.logpdf, np.array([0.5, 0.5]), options=options)
    samples = slicer.sample(20000).samples
    
    assert np.all(np.abs(x - np.mean(samples, axis=0)) < threshold)
