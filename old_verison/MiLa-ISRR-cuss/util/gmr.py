import numpy as np
import scipy.linalg as sp
import scipy.stats as stats

def conditional(mean, cov, x_in, d_in=range(0,1), d_out=range(1,2), N=None):
    nd_in = np.size(d_in)
    nd_out = np.size(d_out)

    if N is None:
        N=np.shape(x_in)
    
    mu_ii = mean[d_in].reshape(nd_in,-1)
    mu_oo = mean[d_out].reshape(nd_out,-1)
    cov_ii = cov[np.ix_(d_in,d_in)]
    prec_ii = np.linalg.inv(cov_ii)
    cov_io = cov[np.ix_(d_in,d_out)]
    cov_oi = cov_io.T
    cov_oo = cov[np.ix_(d_out,d_out)]

    # conditional distribution
    mu_cond = mu_oo.reshape(nd_out,-1) + cov_oi@prec_ii@(x_in-mu_ii).reshape(nd_in,-1)
    mu_cond = mu_cond.T # features as columns
    cov_cond = cov_oo-cov_oi@prec_ii@cov_io
    
    return mu_cond, cov_cond

def gmr(gmm, x_in, d_in=range(0,1), d_out=range(1,2), N=None):
    K = gmm.n_components

    nd_in = np.size(d_in)
    nd_out = np.size(d_out)

    if N is None:
        N = np.shape(x_in)

    # initialize variables to store conditional distributions
    mu_cond = np.zeros((N,len(d_out),K))
    sigma_cond = np.zeros((len(d_out),len(d_out),K)) # doesn't need N because the covariance of the conditional is not input-dep.

    # to store moment matching approximation
    mu = np.zeros((N,len(d_out)))
    sigma = np.zeros((N,len(d_out),len(d_out)))

    h = np.zeros((N,K))

    for i in range(0,gmm.n_components):
        # marginal distribution of the input variable
        mu_ii = gmm.means_[i,np.ix_(d_in)].reshape(nd_in,-1)
        cov_ii = gmm.covariances_[i][np.ix_(d_in,d_in)]

        # conditional distribution for each Gaussian
        mu_cond[:,:,i], sigma_cond[:,:,i] = conditional(gmm.means_[i], gmm.covariances_[i], x_in, d_in, d_out, N)

        # prior update
        h[:,i] = gmm.weights_[i]*stats.multivariate_normal.pdf(np.array(x_in).T, mean=mu_ii.flatten(), cov=cov_ii)

    h = h/np.sum(h,axis=1)[:,None] # priors must sum to 1

    # moment matching
    for i in range(N):
        mu[i,:] = mu_cond[i,:,:]@h[i,:]
        sigma_tmp = np.zeros((nd_out,nd_out))
        for n in range(K):
            sigma_tmp += h[i,n]*(sigma_cond[:,:,n] + np.outer(mu_cond[i,:,n],mu_cond[i,:,n]))
        sigma[i,:,:] = sigma_tmp - np.outer(mu[i,:],mu[i,:])
    return mu, sigma