
import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

import hyperboloid 
import hyperboloid_exponential
from util import *

log_2_pi = np.log(2*np.pi)

class SparseHyperboloidGPLVM(hyperboloid.Hyperboloid):
    """
    Hyperboloid Gaussian process latent variable model
    param:
        X:    latent variables
        sigma: Gaussian noise of lielihood function
    """
    def __init__(self, Y, latent_dim, lengthscale=None, num_inducing=None, X=None, Z=None, init='random'):
        super().__init__(dim=latent_dim)

        if lengthscale is None:
            lengthscale = 5.

        if num_inducing is None:
            num_inducing = 10

        self.num_inducing = num_inducing
        self.Y = Y
        self.data_num, self.data_dim = self.Y.shape
        self.latent_dim = latent_dim
        if X is None:
            X_ = np.random.rand(Y.shape[0], latent_dim) * 1e-3 #* 100
            X_ -= X_.mean()
            X = np.zeros((Y.shape[0], latent_dim+1))
            X[:,1:] = X_
            self.X = self.set_dim0(X)

        self.kern = hyperboloid_exponential.HyperboloidExponential(latent_dim, lengthscale=lengthscale)
        self.sigma = 1. # Gaussian noise parameter (precision)
        self.tr_YYT = np.einsum("ij,ij->", self.Y, self.Y)
        self.update_Z(self.X)

    def RiemannGD(self, lr=None, max_iters=None, min_llf=100):

        if lr is None:
            lr = 1e-02
        if max_iters is None:
            max_iters=1000

        obj = 1e+100
        cur_obj = obj

        for i in tqdm(range(max_iters)):

            param_dict = self.inference(X=self.X, Z=self.Z, variance=self.kern.variance, sigma=self.sigma)

            obj = -1. * param_dict['log_likelihood']

            dX = self.kern.gradients_X_diag(param_dict['dL_dKdiag'], self.X) + \
                 self.kern.gradients_X(param_dict['dL_dKmn'].T, self.X, self.Z)
            dX *= 0.5

            dX_norm = np.sqrt(np.sum(dX**2, axis=1))
            dX_norm = np.where(dX_norm>1., dX_norm, 1.)
            dX = dX / dX_norm[:,None] # normlize if norm > 1
            H = dX.dot(np.diag(self.gl))
            step_X = lr * self.projection(H, self.X)
            cur_X = self.set_dim0(self.exp_map(step_X, self.X))

            dvar = self.kern.dL_dvar_diag(param_dict['dL_dKdiag'], self.X) + \
                   self.kern.dL_dvar(param_dict['dL_dKmn'].T, self.X, self.Z) + \
                   self.kern.dL_dvar(param_dict['dL_dKmm'], self.Z, None)
    
            if np.abs(dvar) >= 1.:
                dvar /= np.abs(dvar)
            cur_variance = max(self.kern.variance + lr * dvar, 1e-8)

            dsigma = param_dict['dL_dsigma']

            if np.abs(dsigma) >= 1.:
                dsigma /= np.abs(dsigma)
            cur_sigma =  max(self.sigma + lr * dsigma, 1e-8)

            variance = self.kern.variance
            self.kern.variance = cur_variance
            cur_obj = -1. * self.inference(X=cur_X, Z=self.Z, variance=cur_variance, sigma=cur_sigma, only_likelihood=True)

            if cur_obj < obj:
                self.X = cur_X
                self.kern.variance = cur_variance
                self.sigma = cur_sigma
            else:
                self.kern.variance = variance
                lr *= 0.1

            if i % 100 == 0:
                print('%d-th iter: mean norm:%.5lf, obj:%.5lf'%(i, np.mean(dX**2), cur_obj))

            if i > 0 and i % 5 == 0:
                self.update_Z(self.X)
                lr = 1e-3

            if (i > 200) and (obj < min_llf):
                break

    def inference(self, X, Z, variance, sigma, only_likelihood=False):
        """
        wrapper function
        X : latent variables
        Z : inducing points (not parameter)
        Kmm, Kmmi : gram matrix of Z and its inverse matrix
        Lm : Cholesky decomposition of Kmm
        variance : kernel parameter
        sigma : variance parameter
        """
        Kmn = self.kern.K(X=Z, X2=X)
        Kmm = self.kern.K(X=Z) # + 1e-8 * np.eye(self.num_inducing)
        Lm = self.Kchol(Kmm)
        Kmmi = self.Kinv(Lm)
        precision = 1. / max(sigma, 1e-8)
        A = Kmm + precision * Kmn.dot(Kmn.T)
        LA = self.Kchol(A)
        Ai = self.Kinv(LA)
        data_fit = precision * Ai.dot(Kmn.dot(self.Y))
        log_likelihood = self.log_likelihood(A, Ai, LA, Kmm, Kmmi, Lm, Kmn, data_fit, variance, precision)
        if only_likelihood:
            return log_likelihood

        dL_dKmm = self.dL_dKmm(Ai, Kmmi, Kmn, data_fit, precision)
        dL_dKmn = self.dL_dKmn(Ai, Kmmi, Kmn, data_fit, precision)
        dL_dsigma = self.dL_dsigma(Ai, Kmmi, Kmn, data_fit, variance, precision)
        dL_dKdiag = -0.5 * self.data_dim * (precision * np.ones([self.data_num, 1])).flatten()

        param_dict = {
            'log_likelihood':log_likelihood,
            'dL_dKmm':dL_dKmm,
            'dL_dKmn':dL_dKmn,
            'dL_dsigma':dL_dsigma,
            'dL_dKdiag':dL_dKdiag
            }

        return param_dict

    def log_likelihood(self, A, Ai, LA, Kmm, Kmmi, Lm, Kmn, data_fit, variance, precision):
        """
        compute log likelihood from Cholesky factor
        """
        lik1 = 0.5 * self.data_num * self.data_dim * (np.log(precision) - log_2_pi) # const. term
        lik2 = self.data_dim * (np.sum(np.log(np.diag(Lm))) - np.sum(np.log(np.diag(LA)))) # log determinant term
        tmp = precision * Kmn.dot(self.Y)
        lik3 = 0.5 * (np.sum(data_fit * tmp) - precision * self.tr_YYT) # data fit term (i.e., with observation Y)
        lik4 = -0.5 * precision * self.data_dim * (self.data_num * variance - np.sum(Kmmi * (Kmn.dot(Kmn.T)))) # trace term

        return lik1 + lik2 + lik3 + lik4

    def dL_dKmm(self, Ai, Kmmi, Kmn, data_fit, precision):
        """
        compute dL_dKmm
        """
        dL_dKmm = 0.5 * self.data_dim * (Kmmi - Ai)  # w.r.t. log determinant term
        dL_dKmm += -0.5 * np.dot(data_fit, data_fit.T) # w.r.t. data fit term
        tmp = Kmmi.dot(Kmn)
        dL_dKmm += -0.5 * precision * self.data_dim * np.dot(tmp, tmp.T) # w.r.t. trace term

        return dL_dKmm

    def dL_dKmn(self, Ai, Kmmi, Kmn, data_fit, precision):
        """
        compute dL_dKmn
        """
        dL_dKmn = -1. * self.data_dim * precision * Ai.dot(Kmn) # w.r.t. log determinant term
        dL_dKmn += precision * data_fit.dot(self.Y.T) - precision * (np.dot(data_fit, data_fit.T)).dot(Kmn) # w.r.t. data fit term
        dL_dKmn += precision * self.data_dim * Kmmi.dot(Kmn) # w.r.t. trace term

        return dL_dKmn

    def dL_dsigma(self, Ai, Kmmi, Kmn, data_fit, variance, precision):
        """
        compute gradient of Gaussian noise
        """
        Kmn_Knm = Kmn.dot(Kmn.T)
        dL_dsigma = -0.5 * self.data_num * self.data_dim * precision # w.r.t. const. term
        dL_dsigma += 0.5 * self.data_dim * np.sum(Ai * Kmn_Knm) * precision**2 # w.r.t. log determinant term
        tmp = Kmn.dot(self.Y)
        dL_dsigma += 0.5 * precision**2 * self.tr_YYT - precision**2 * np.sum(data_fit * tmp) +\
                     0.5 * precision**2 * np.sum(Kmn_Knm * (np.dot(data_fit, data_fit.T))) # w.r.t. data fit term
        dL_dsigma += 0.5 * self.data_dim * precision**2 * (self.data_num * variance - np.sum(Kmmi * Kmn_Knm)) # w.r.t. trace term
        
        return dL_dsigma


    def update_Z(self, X):
        """
        update Z (active sets)
        """
        sample_indices = sorted(random.sample(range(X.shape[0]), self.num_inducing))
        self.Z = self.set_dim0(X[sample_indices, :])#  + 1e-05 * np.random.rand(self.num_inducing, X.shape[1]))

    def Kchol(self, K):
        """
        compute Cholesky factor
        """
        return jitchol(K)
        
    def Kinv(self, L):
        """
        compute inverse matrix from Cholesky factor
        """
        L = np.asfortranarray(L)
        Ki, _ = lapack.dpotri(L, lower = True)
        symmetrify(Ki)
        return Ki

    def check_grad_var(self):
        ep = 1e-05
        var = self.kern.variance
        param_dict = self.inference(self.X, self.Z, var, self.sigma, only_likelihood=False)
        self.kern.variance = var - ep
        lik1 = self.inference(self.X, self.Z, var-ep, self.sigma, only_likelihood=True)
        self.kern.variance = var + ep
        lik2 = self.inference(self.X, self.Z, var+ep, self.sigma, only_likelihood=True)
        self.kern.variance = var

        dvar = self.kern.dL_dvar_diag(param_dict['dL_dKdiag'], self.X) + \
                self.kern.dL_dvar(param_dict['dL_dKmn'].T, self.X, self.Z) + \
                self.kern.dL_dvar(param_dict['dL_dKmm'], self.Z, None)

        print('Variance  analytic value: %.5lf  numerical value: %.5lf'%(dvar, (lik2-lik1) / (2*ep)))

    def check_grad_sigma(self):
        ep = 1e-05
        sigma = self.sigma
        param_dict = self.inference(self.X, self.Z, self.kern.variance, sigma, only_likelihood=False)
        lik1 = self.inference(self.X, self.Z, self.kern.variance, sigma-ep, only_likelihood=True)
        lik2 = self.inference(self.X, self.Z, self.kern.variance, sigma+ep, only_likelihood=True)

        dsigma = param_dict['dL_dsigma']

        print('Gaussian noise  analytic value: %.5lf  numerical value: %.5lf'%(dsigma, (lik2-lik1) / (2*ep)))

    def check_grad_X(self):
        ep = 1e-5
        X = self.X
        X_ = X
        param_dict = self.inference(X, self.Z, self.kern.variance, self.sigma, only_likelihood=False)
        dX = self.kern.gradients_X_diag(param_dict['dL_dKdiag'], X) + self.kern.gradients_X(param_dict['dL_dKmn'].T, X, self.Z)
        dX *= 0.5

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X = X_
                X[i][j] += ep
                X = self.set_dim0(X)
                lik1 = self.inference(X, self.Z, self.kern.variance, self.sigma, only_likelihood=True)
                X = X_
                X[i][j] -= ep
                X = self.set_dim0(X)
                lik2 = self.inference(X, self.Z, self.kern.variance, self.sigma, only_likelihood=True)
                print('latent variables[%d][%d] analytic value: %.5lf  numerical value: %.5lf'%(i, j, dX[i][j], (lik2-lik1) / (2*ep)))

    def reconst(self):
        Kmn = self.kern.K(self.Z, self.X)
        Kmm = self.kern.K(self.Z, self.Z)
        A = Kmm + Kmn.dot(Kmn.T) / self.sigma
        Ai = self.Kinv(self.Kchol(A))
        Qi = np.eye(self.data_num) / self.sigma - Kmn.T.dot(Ai).dot(Kmn) / (self.sigma**2)
        return self.kern.K(self.X, self.X).dot(Qi).dot(self.Y)