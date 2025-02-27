
import numpy as np
import random
from tqdm import tqdm_notebook as tqdm

import hyperboloid
import hyperboloid_exponential
from util import *

log_2_pi = np.log(2*np.pi)

class BayesianHyperboloidGPLVM(hyperboloid.Hyperboloid):
    """
    Hyperboloid Gaussian process latent variable model
    param:
        X:    latent variables
        sigma: Gaussian noise of likelihood function
    """
    def __init__(self, Y, latent_dim, lengthscale=None, num_inducing=None, sampling_num=None, M=None, Z=None):
        super().__init__(dim=latent_dim)

        if lengthscale is None:
            lengthscale = 5.
        if num_inducing is None:
            num_inducing = 10
        if sampling_num is None:
            sampling_num = 10

        self.num_inducing = num_inducing
        self.sampling_num = sampling_num
        self.Y = Y
        self.KL_factor = 1.
        self.data_num, self.data_dim = self.Y.shape
        self.latent_dim = latent_dim
        if M is None:
            M_ = np.random.rand(Y.shape[0], latent_dim) * 1e-02
            M_ -= M_.mean()
            M = np.zeros((Y.shape[0], latent_dim+1))
            M[:,1:] = M_
            self.M = self.set_dim0(M)

        self.S = np.random.rand(Y.shape[0], latent_dim) * 1e-05
        self.mu0 = np.zeros(self.latent_dim+1)
        self.mu0[0] = 1.

        self.kern = hyperboloid_exponential.HyperboloidExponential(latent_dim, lengthscale=lengthscale)
        self.sigma = 1. # Gaussian noise parameter (precision)
        self.tr_YYT = np.einsum("ij,ij->", self.Y, self.Y)
        self.update_Z(self.M)
        self.seed = np.random.randint(1)


    def RiemannGD(self, lr=None, max_iters=None, min_llf=100):

        if lr is None:
            lr = 1e-02
        if max_iters is None:
            max_iters=1000

        for i in tqdm(range(max_iters)):

            self.seed = np.random.randint(1)
            param_dict = self.inference(M=self.M, S=self.S, Z=self.Z, variance=self.kern.variance, sigma=self.sigma, seed=self.seed)

            obj = -1. * param_dict['log_likelihood']
            KL = param_dict['KL_divergence']

            # project dM on the surface
            dM = param_dict['dL_dM']
            dM_norm = np.sqrt(np.sum(dM**2, axis=1))
            dM_norm = np.where(dM_norm>1., dM_norm, 1.)
            dM = dM / dM_norm[:,None]
            H = dM.dot(np.diag(self.gl))
            step_M = lr * self.projection(H, self.M)
            cur_M = self.set_dim0(self.exp_map(step_M, self.M)) # compute current position

            dS = param_dict['dL_dS']
            dS = np.where(np.abs(dS)>1., dS / np.abs(dS), dS)
            cur_S = self.S + lr * dS
            cur_S = np.where(cur_S<=1e-08, 1e-08, cur_S)

            dvar = param_dict['dL_dvar']
            if np.abs(dvar) >= 1.:
                dvar /= np.abs(dvar)
            cur_variance = max(self.kern.variance + lr * dvar, 1e-8)

            dsigma = param_dict['dL_dsigma']
            if np.abs(dsigma) >= 1.:
                dsigma /= np.abs(dsigma)
            cur_sigma =  max(self.sigma + lr * dsigma, 1e-8)

            variance = self.kern.variance
            self.kern.variance = cur_variance
            cur_obj = -1. * self.inference(cur_M, cur_S, Z=self.Z, variance=cur_variance, sigma=cur_sigma,seed=self.seed, only_likelihood=True)

            if cur_obj < obj:
                self.M = cur_M
                self.kern.variance = cur_variance
                self.sigma = cur_sigma
                if i >= 100:
                    self.S = cur_S
            else:
                self.kern.variance = variance
                lr *= 0.1

            if i % 100 == 0:
                print('%d-th iter: obj:%.5lf, KL:%.5lf, beta:%.5lf, variance:%.5lf, mean of S:%.5lf'%(i, obj, self.KL_factor * KL, 1./self.sigma, self.kern.variance, self.S.mean()))

            if i > 0 and i % 5 == 0:
                self.update_Z(self.M)
                lr = 1e-3

            if (i > 200) and (np.abs(cur_obj-obj) <= 1e-5 or obj < min_llf):
                break


    def inference(self, M, S, Z, variance, sigma, seed, only_likelihood=False):
        """
        wrapper function
        X : latent variables
        Z : inducing points (not parameter)
        Kmm, Kmmi : gram matrix of Z and its inverse matrix
        Lm : Cholesky decomposition of Kmm
        variance : kernel parameter
        sigma : variance parameter
        """
        # estimating psi statics and KL divergence based on Monte Carlo method

        psi1 = np.zeros((self.num_inducing, self.data_num))
        psi2 = np.zeros((self.num_inducing, self.num_inducing))
        KL = 0.
        np.random.seed(seed)
        zeta = np.random.randn(self.sampling_num, self.data_num, self.latent_dim)

        for i in range(self.sampling_num):
            KL_, X_ = self.sampling_X_and_KL(zeta[i,:,:], M, S)
            tmp = self.kern.K(X=Z, X2=X_)
            psi1 += tmp
            psi2 += tmp.dot(tmp.T)
            KL += KL_

        psi1 /= self.sampling_num
        psi2 /= self.sampling_num
        KL /= self.sampling_num

        Kmm = self.kern.K(X=Z)
        Lm = self.Kchol(Kmm)
        Kmmi = self.Kinv(Lm)
        precision = 1. / max(sigma, 1e-8)
        A = Kmm + precision * psi2
        LA = self.Kchol(A)
        Ai = self.Kinv(LA)
        data_fit = precision * Ai.dot(psi1.dot(self.Y))
        log_likelihood = self.log_likelihood(LA, Kmmi, Lm, psi1, psi2, data_fit, variance, precision) - self.KL_factor * KL

        if only_likelihood:
            return log_likelihood

        # estimating each derivatives
        dL_dM = np.zeros(M.shape)
        dL_dS = np.zeros(S.shape)
        dL_dpsi1 = self.dL_dpsi1(data_fit, precision)
        dL_dpsi2 = self.dL_dpsi2(Kmmi, Ai, data_fit, precision)
        dL_dKmm = self.dL_dKmm(Ai, Kmmi, psi2, data_fit, precision)

        for i in range(self.sampling_num):
            _, X_ = self.sampling_X_and_KL(zeta[i,:,:], M, S)
            dL_dX = self.kern.dL_dX(dL_dpsi1.T, X_, Z) + self.kern.dL_dX_via_psi2(dL_dpsi2, X_, Z)
            dL_dM += self.dL_dM(dL_dX, M, S, zeta[i,:,:]) - self.KL_factor * self.dKL_dM(M, S, zeta[i,:,:])
            dL_dS += self.dL_dS(dL_dX, M, S, zeta[i,:,:]) - self.KL_factor * self.dKL_dS(M, S, zeta[i,:,:])
        dL_dM /= self.sampling_num
        dL_dS /= self.sampling_num
        dL_dsigma = self.dL_dsigma(Ai, Kmmi, psi1, psi2, data_fit, variance, precision)
        dL_dvar = self.kern.dL_dvar(dL_dKmm, Z, None) + \
                  self.kern.dL_dvar_via_psi0(self.data_num, self.data_dim, precision) + \
                  self.kern.dL_dvar_via_psi1(dL_dpsi1, psi1) + self.kern.dL_dvar_via_psi2(dL_dpsi2, psi2)

        param_dict = {
            'log_likelihood':log_likelihood,
            'KL_divergence':KL,
            'dL_dM':dL_dM,
            'dL_dS':dL_dS,
            'dL_dsigma':dL_dsigma,
            'dL_dvar':dL_dvar
            }

        return param_dict


    def sampling_X_and_KL(self, zeta, M, S):
        V = S * zeta
        U = self.parallel_transport(np.concatenate([np.zeros(self.data_num).reshape(1,-1).T, V], 1), M, self.mu0[None,:])
        norm_U = np.sqrt(self.Ldot_diag(U))
        norm_V = np.sqrt(np.sum(V**2, axis=1))
        f_1 = np.divide(np.sinh(norm_U) , norm_U, out=np.zeros(norm_U.shape), where=(norm_U!=0))
        f_2 = np.divide(np.sinh(norm_V) , norm_V, out=np.zeros(norm_V.shape), where=(norm_V!=0))
        KL_1 = -0.5 * self.data_num * self.latent_dim * log_2_pi - np.sum(np.log(S)) - 0.5 * np.sum(zeta**2) - (self.latent_dim-1) * np.sum(np.log(f_1))
        KL_2 = -0.5 * self.data_num * self.latent_dim * log_2_pi - 0.5 * np.sum(V*V) - (self.latent_dim-1) * np.sum(np.log(f_2))
        KL = KL_1 - KL_2

        return KL, self.exp_map(U, M)


    def log_likelihood(self, LA, Kmmi, Lm, psi1, psi2, data_fit, variance, precision):
        """
        compute log likelihood from Cholesky factor
        """
        lik1 = 0.5 * self.data_num * self.data_dim * (np.log(precision) - log_2_pi) # const. term
        lik2 = self.data_dim * (np.sum(np.log(np.diag(Lm))) - np.sum(np.log(np.diag(LA)))) # log determinant term
        tmp = precision * psi1.dot(self.Y)
        lik3 = 0.5 * (np.sum(data_fit * tmp) - precision * self.tr_YYT) # data fit term (i.e., with observation Y)
        lik4 = -0.5 * precision * self.data_dim * (self.data_num * variance - np.sum(Kmmi * psi2)) # trace term

        return lik1 + lik2 + lik3 + lik4


    def dL_dKmm(self, Ai, Kmmi, psi2, data_fit, precision):
        """
        compute dL_dKmm
        """
        dL_dKmm = 0.5 * self.data_dim * (Kmmi - Ai)  # w.r.t. log determinant term
        dL_dKmm -= 0.5 * np.dot(data_fit, data_fit.T) # w.r.t. data fit term
        dL_dKmm -= 0.5 * precision * self.data_dim * Kmmi.dot(psi2).dot(Kmmi) # w.r.t. trace term

        return dL_dKmm


    def dL_dpsi1(self, data_fit, precision):

        return precision * data_fit.dot(self.Y.T)


    def dL_dpsi2(self, Kmmi, Ai, data_fit, precision):

        return 0.5 * precision * self.data_dim * (Kmmi - Ai) - 0.5 * precision * data_fit.dot(data_fit.T)


    def dL_dM(self, dL_dX, M, S, zeta):

        V = S * zeta
        U = self.parallel_transport(np.concatenate([np.zeros(self.data_num).reshape(1,-1).T, V], 1), M, self.mu0[None,:])
        norm_U = np.sqrt(self.Ldot_diag(U))
        dL_dM = np.zeros(M.shape)  # (N, Q+1)

        for q in range(self.latent_dim+1):
            dUnorm_dMq = self.dUnorm_dMq(U, M, V, q)

            dX_dM = np.sinh(norm_U)[:,None] * dUnorm_dMq[:,None] * M # second term
            tmp = (np.cosh(norm_U) * norm_U - np.sinh(norm_U)) / (norm_U**2)
            dX_dM += tmp[:,None] * dUnorm_dMq[:,None] * U # third term
            dU_dMq = self.dU_dMq(M, V, q)
            dX_dM += np.sinh(norm_U)[:,None] / norm_U[:,None] * dU_dMq # fourth term
            dX_dM[:,q] += np.cosh(norm_U) # first term

            dL_dM[:,q] = np.sum(dL_dX * dX_dM, axis=1)

        return dL_dM


    def dL_dS(self, dL_dX, M, S, zeta):

        V = S * zeta
        U = self.parallel_transport(np.concatenate([np.zeros(self.data_num).reshape(1,-1).T, V], 1), M, self.mu0[None,:])
        norm_U = np.sqrt(self.Ldot_diag(U))
        dL_dS = np.zeros(S.shape)  # (N, Q)

        for q in range(self.latent_dim):
            dUnorm_dSq = self.dUnorm_dSq(U, M, zeta, q)

            tmp = (np.cosh(norm_U) * norm_U - np.sinh(norm_U)) / (norm_U**2)
            dX_dS = np.sinh(norm_U)[:,None] * dUnorm_dSq[:,None] * M + \
                    tmp[:,None] * dUnorm_dSq[:,None] * M  # first and second term
            dU_dSq = self.dU_dSq(M, zeta, q)
            dX_dS += np.sinh(norm_U)[:,None] / norm_U[:,None] * dU_dSq # third term

            dL_dS[:,q] = np.sum(dL_dX * dX_dS, axis=1)

        return dL_dS


    def dKL_dM(self, M, S, zeta):

        V = S * zeta
        U = self.parallel_transport(np.concatenate([np.zeros(self.data_num).reshape(1,-1).T, V], 1), M, self.mu0[None,:])
        norm_U = np.sqrt(self.Ldot_diag(U))
        dKL_dM = np.zeros(M.shape)

        for q in range(M.shape[1]):
            dUnorm_dMq = self.dUnorm_dMq(U, M, V, q)
            dKL_dM[:,q] = (self.latent_dim-1) * dUnorm_dMq / norm_U - (self.latent_dim-1) * dUnorm_dMq / np.tanh(norm_U)

        return dKL_dM


    def dKL_dS(self, M, S, zeta):

        V = S * zeta
        U = self.parallel_transport(np.concatenate([np.zeros(self.data_num).reshape(1,-1).T, V], 1), M, self.mu0[None,:])
        norm_U = np.sqrt(self.Ldot_diag(U))
        norm_V = np.sqrt(np.sum(V**2, axis=1))
        dVnorm_dS = self.dVnorm_dS(S, V, zeta)
        dKL_dS = zeta * zeta * S - 1. / S + (self.latent_dim - 1) * dVnorm_dS / np.tanh(norm_V)[:,None] - (self.latent_dim - 1) * dVnorm_dS / norm_V[:,None]

        for q in range(self.latent_dim):
            dUnorm_dSq = self.dUnorm_dSq(U, M, zeta, q)
            dKL_dS[:,q] += (self.latent_dim-1) * dUnorm_dSq / norm_U - (self.latent_dim-1) * dUnorm_dSq / np.tanh(norm_U)

        return dKL_dS


    def dUnorm_dMq(self, U, M, V, q):

        U_tilde = U
        U_tilde[:,0] *= -1.
        dU_dMq = self.dU_dMq(M, V, q)

        return np.sum(U_tilde * dU_dMq, axis = 1) / np.sqrt(self.Ldot_diag(U)) # (N,)


    def dU_dMq(self, M, V, q):

        M_dot_V = np.sum(M[:,1:] * V, axis = 1)
        if q == 0:
            dU_dMq = -1. * M_dot_V[:,None] / ((M[:,0][:,None] + 1.)**2) * (self.mu0[None,:] + M)
        else:
            dU_dMq = V[:,q-1][:,None] / (M[:,0][:,None] + 1.) * (self.mu0[None,:] + M)
        dU_dMq[:,q] += M_dot_V / (M[:,0] + 1.)

        return dU_dMq # (N, Q+1)


    def dUnorm_dSq(self, U, M, zeta, q):

        U_tilde = U
        U_tilde[:,0] *= -1
        dU_dSq = self.dU_dSq(M, zeta, q)

        return  np.sum(U_tilde * dU_dSq, axis=1) / np.sqrt(self.Ldot_diag(U))  #(N,)


    def dVnorm_dS(self, S, V, zeta):
        
        return S * zeta**2 / np.sqrt((np.sum(V**2, axis=1)))[:,None]

    def dU_dSq(self, M, zeta, q):

        dU_dSq = M[:,q+1][:,None] * zeta[:,q][:,None] / (M[:,0][:,None] + 1.) * (self.mu0[None,:] + M)
        dU_dSq[:,q+1] += zeta[:,q]

        return dU_dSq  #(N, Q)


    def dL_dsigma(self, Ai, Kmmi, psi1, psi2, data_fit, variance, precision):
        """
        compute gradient of Gaussian noise
        """
        dL_dsigma = -0.5 * self.data_num * self.data_dim * precision # w.r.t. const. term
        dL_dsigma += 0.5 * self.data_dim * np.sum(Ai * psi2) * precision**2 # w.r.t. log determinant term
        tmp = psi1.dot(self.Y)
        dL_dsigma += 0.5 * precision**2 * self.tr_YYT - precision**2 * np.sum(data_fit * tmp) +\
                     0.5 * precision**2 * np.sum(psi2 * (np.dot(data_fit, data_fit.T))) # w.r.t. data fit term
        dL_dsigma += 0.5 * self.data_dim * precision**2 * (self.data_num * variance - np.sum(Kmmi * psi2)) # w.r.t. trace term

        return dL_dsigma


    def update_Z(self, M):
        """
        update Z (active sets)
        """
        sample_indices = sorted(random.sample(range(M.shape[0]), self.num_inducing))
        self.Z = self.set_dim0(M[sample_indices, :]) #+ 1e-05 * np.random.rand(self.num_inducing, self.M.shape[1])


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
        param_dict = self.inference(self.M, self.S, self.Z, var, self.sigma, self.seed, only_likelihood=False)
        self.kern.variance = var - ep
        lik1 = self.inference(self.M, self.S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
        self.kern.variance = var + ep
        lik2 = self.inference(self.M, self.S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
        self.kern.variance = var

        dvar = param_dict['dL_dvar']

        print('Variance  analytic value: %.5lf  numerical value: %.5lf'%(dvar, (lik2-lik1) / (2*ep)))

    def check_grad_sigma(self):
        ep = 1e-05
        sigma = self.sigma
        param_dict = self.inference(self.M, self.S, self.Z, self.kern.variance, sigma, self.seed, only_likelihood=False)
        lik1 = self.inference(self.M, self.S, self.Z, self.kern.variance, sigma-ep, self.seed, only_likelihood=True)
        lik2 = self.inference(self.M, self.S, self.Z, self.kern.variance, sigma+ep, self.seed, only_likelihood=True)

        dsigma = param_dict['dL_dsigma']

        print('Gaussian noise  analytic value: %.5lf  numerical value: %.5lf'%(dsigma, (lik2-lik1) / (2*ep)))


    def check_grad_M(self):
        ep = 1e-5
        M = self.M
        M_ = M
        param_dict = self.inference(M, self.S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=False)
        dM = param_dict['dL_dM']
        dM *= 0.5

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M = M_
                M[i][j] += ep
                M = self.set_dim0(M)
                lik1 = self.inference(M, self.S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
                M = M_
                M[i][j] -= ep
                M = self.set_dim0(M)
                lik2 = self.inference(M, self.S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
                print('latent variables[%d][%d] analytic value: %.5lf  numerical value: %.5lf'%(i, j, dM[i][j], (lik1-lik2) / (2*ep)))

    def check_grad_S(self):
        ep = 1e-5
        S = self.S
        S_ = S
        param_dict = self.inference(self.M, S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=False)
        dS = param_dict['dL_dS']
        dS *= 0.5

        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                S = S_
                S[i][j] += ep
                lik1 = self.inference(self.M, S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
                S = S_
                S[i][j] -= ep
                lik2 = self.inference(self.M, S, self.Z, self.kern.variance, self.sigma, self.seed, only_likelihood=True)
                print('covariance S[%d][%d] analytic value: %.5lf  numerical value: %.5lf'%(i, j, dS[i][j], (lik1-lik2) / (2*ep)))
