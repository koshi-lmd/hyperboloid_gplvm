
import numpy as np
from tqdm import tqdm_notebook as tqdm

import hyperboloid 
import hyperboloid_exponential
from util import *

log_2_pi = np.log(2*np.pi)


class HyperboloidGPLVM(hyperboloid.Hyperboloid):
    """
    Hyperboloid Gaussian process latent variable model
    param:
        X:    latent variables
        sigma: Gaussian noise of lielihood function
    """
    def __init__(self, Y, latent_dim, X=None, lengthscale=None):
        super().__init__(dim=latent_dim)
        self.Y = Y
        if X is None:
            X_ = np.random.rand(Y.shape[0], latent_dim) * 1e-03
            X_ -= X_.mean()
            X = np.zeros((Y.shape[0], latent_dim+1))
            X[:,1:] = X_
            self.X = self.set_dim0(X)

        if lengthscale is None:
            lengthscale = 10.

        self.kern = hyperboloid_exponential.HyperboloidExponential(latent_dim, lengthscale=lengthscale)
        self.sigma = 1. # Gaussian noise parameter

    def RiemannGD(self, lr=None, max_iters=None):

        if lr is None:
            lr = 1e-02
        if max_iters is None:
            max_iters=300

        obj = 1e+100
        cur_obj = obj
        L = None

        for i in tqdm(range(max_iters)):

            K = self.kern.K(self.X) + (self.sigma + 1e-8) * np.eye(self.Y.shape[0])
            param_dict = self.inference(K, L=L)

            obj = -1. * param_dict['log_likelihood']

            dX = self.kern.gradients_X(param_dict['dL_dK'], self.X, None)
            dX_norm = np.sqrt(np.sum(dX**2, axis=1))
            dX_norm = np.where(dX_norm>1., dX_norm, 1.)
            dX = dX / dX_norm[:,None] # normlize if norm > 1
            H = dX.dot(np.diag(self.gl))
            cur_X = self.exp_map(lr * self.projection(H, self.X), self.X)
            cur_X = self.set_dim0(cur_X)

            dvar = self.kern.dL_dvar(param_dict['dL_dK'], self.X, None)
            if np.abs(dvar) >= 1.:
                dvar /= np.abs(dvar)
            cur_variance = max(self.kern.variance + lr * dvar, 1e-10)

            dsigma = self.dL_dsigma(param_dict['dL_dK'])
            if np.abs(dsigma) >= 1.:
                dsigma /= np.abs(dsigma)
            cur_sigma = max(self.sigma + lr * dsigma, 1e-10)

            variance = self.kern.variance
            self.kern.variance = cur_variance

            K = self.kern.K(cur_X) + cur_sigma * np.eye(self.Y.shape[0])
            cur_obj = -1. * self.inference(K, only_likelihood=True)

            if cur_obj < obj:
                self.X = cur_X
                self.kern.variance = cur_variance
                self.sigma = cur_sigma
                L = None
            else:
                self.kern.variance = variance
                L = param_dict['Cholesky']
                lr *= 0.1
            
            if i % 10 == 0:
                print('%d-th iter: mean norm:%.5lf, obj:%.5lf'%(i, np.mean(dX**2), cur_obj))

            if np.abs(cur_obj-obj) <= 1e-5 or obj < 100.:
                break

    def inference(self, K, L=None, only_likelihood=False):
        """
        wrapper function
        """
        if L is None:
            L = self.Kchol(K)
        Ki = self.Kinv(L)
        log_likelihood = self.log_likelihood(L, Ki)
        if only_likelihood:
            return log_likelihood
        dL_dK = self.dL_dK(L, Ki)

        param_dict = {
            'log_likelihood':log_likelihood,
            'Cholesky':L,
            'dL_dK':dL_dK
            }
        
        return param_dict

    def Kchol(self, K):
        """
        compute Cholesky factor
        """
        return jitchol(K)

    def log_likelihood(self, L, Ki):
        """
        compute log likelihood from Cholesky factor
        """
        logdet = 2. * np.sum(np.log(np.diag(L)))
        Ki = self.Kinv(L)

        return 0.5*(-self.Y.size * log_2_pi - self.Y.shape[1] * logdet - np.sum((Ki.dot(self.Y)) * self.Y))

    def Kinv(self, L):
        """
        compute inverse matrix from Cholesky factor
        """
        L = np.asfortranarray(L)
        Ki, _ = lapack.dpotri(L, lower = True)
        symmetrify(Ki)
        return Ki
    
    def dL_dK(self, L, Ki):
        """
        compute dL_dK from Cholesky factor
        """
        alpha = Ki.dot(self.Y)

        return 0.5 * (np.dot(alpha, alpha.T) - self.Y.shape[1] * Ki)

    def dL_dsigma(self, dL_dK):
        """
        compute gradient of Gaussian noise
        """
        return np.diag(dL_dK).sum()
