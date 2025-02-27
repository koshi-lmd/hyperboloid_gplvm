
import hyperboloid
from paramz.caching import Cache_this
import numpy as np

class HyperboloidExponential(hyperboloid.Hyperboloid):

    def __init__(self, latent_dim, variance=1., lengthscale=5.):
        super().__init__(dim=latent_dim)
        self.latent_dim = latent_dim
        self.variance = variance
        self.lengthscale = lengthscale

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        r = self.distance(X, X2)
        return self.variance * np.exp(-r/(self.lengthscale))

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        """Compute the diagonal of the covariance matrix for X."""
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def gradients_X(self, dL_dK, X, X2 = None):
        """Derivative of the covariance matrix with respect to X"""
        if X2 is None:
            return 2 * self.dL_dX(dL_dK, X, X2)
        else:
            return self.dL_dX(dL_dK, X, X2)

    def gradients_X_diag(self, dL_dKdiag, X):
        return np.zeros(X.shape)

    def dL_dvar(self, dL_dK, X, X2):
        return (dL_dK*self.K(X, X2)).sum() / self.variance

    def dL_dvar_diag(self, dL_dKdiag, X):
        return np.sum(dL_dKdiag)

    def dL_dl(self, dL_dK, X, X2):
        dK_dl = self.distance(X, X2) * self.K(X, X2) / (self.lengthscale**2)
        return (dL_dK * dK_dl).sum()
    
    def dL_dl_diag(self, dL_dKdiag, X):
        return 0.

    @Cache_this(limit=3, ignore_args=())
    def dL_dX(self, dL_dK, X, X2):
        if X2 is None:
            X2 = X
        S = self.Ldot(X, X2)
        S[S>=-1.] = -1.
        f = np.sqrt(S**2. - 1.)
        A = np.divide(self.K(X, X2), f, out=np.zeros(f.shape), where=(f!=0.))
        grad = np.zeros(X.shape)
        X_0 = X[:,0]
        X2_0 = X2[:,0]
    
        for q in range(X.shape[1]):
            tmp = A * (X2[:,q][None,:] - X[:,q][:,None] / X_0[:,None] * X2_0[None,:]) / (self.lengthscale)
            grad[:,q] = np.sum(dL_dK * tmp, axis=1)

        return grad

######### for Bayesian inference #############################################################

    @Cache_this(limit=3, ignore_args=())
    def dL_dX_via_psi2(self, dL_dpsi2, X, X2):
        """
        for Bayesian model
        """
        # factor1 = self.dL_dX(dL_dpsi1, X, X2) # containing psi1
        S = self.Ldot(X, X2)
        S[S>=-1.] = -1.
        f = np.sqrt(S**2. - 1.)
        A = np.divide(self.K(X, X2), f, out=np.zeros(f.shape), where=(f!=0.))
        grad = np.zeros(X.shape)
        X_0 = X[:,0]
        X2_0 = X2[:,0]
        Knm = self.K(X, X2)

        for q in range(X.shape[1]):
            tmp = A * (X2[:,q][None,:] - X[:,q][:,None] / X_0[:,None] * X2_0[None,:]) / (self.lengthscale) # (N, M)
            # tmp = tmp.T
            # print(tmp.shape)
            dpsi2_dXq = tmp[:,None,:] * Knm[:,:,None] + tmp[:,:,None] * Knm[:,None,:] #(N, M, M)
            grad[:,q] = (dL_dpsi2[None,:,:] * dpsi2_dXq).sum(2).sum(1) #/ 2.

        return grad

    def dL_dvar_via_psi2(self, dL_dpsi2, psi2):
        return 2 * (dL_dpsi2 * psi2).sum() / self.variance

    def dL_dvar_via_psi1(self, dL_dpsi1, psi1):
        return (dL_dpsi1 * psi1).sum() / self.variance

    def dL_dvar_via_psi0(self, data_num, data_dim, precision):
        return -0.5 * data_num * data_dim * precision
