
import numpy as np

class Hyperboloid:
    def __init__(self, dim):
        self.gl = np.ones(dim+1)
        self.gl[0] = -1
    
    def set_dim0(self, X):
        """
        """
        X[:,0] = np.sqrt(1. + np.sum(X[:,1:]**2, axis=1))
        return X

    def exp_map(self, X, mu):
        """
        exponential mapping from tangent space to Hyperboloid model
        exp_{mu}(X)
        X:(N, Q) mu:(N, Q)
        return (N, Q) matrix whose (i)-th row vector is exp_{mu[i]}(X[i])
        """        
        norm = np.sqrt(self.Ldot_diag(X))
        return np.cosh(norm)[:,None] * mu + np.sinh(norm)[:,None] / norm[:,None] * X

    def projection(self, X, mu):
        """
        orthogonal projection from real (n+1)-space R^{n+1} to tangent space TL^{n}
        proj_{mu}(X)
        X:(N, Q) mu:(N, Q)
        return (N, Q) matrix whose (i)-th row vector is proj_{mu[i]}(X[i])
        """
        w = self.Ldot_diag(mu, X)
        return X + w[:,None] * mu
    
    def Ldot_diag(self, X, X2=None):
        """
        norm or entry-wise dot product in Hyperboloid model
        X:(N, Q) X2:(N, Q)
        return (N,) vector whose (i,)-th entry is <X[i], X2[i]>_{L}
        """
        if X2 is None:
            tmp = X**2
        else:
            tmp = X * X2
        return np.sum(tmp[:,1:], axis=1) - tmp[:,0]

    def Ldot(self, X, X2=None):
        """
        dot product in Hyperboloid (Lorentzian scalar product)
        X:(N, Q), X2:(M, Q)
        return (N, M) matrix whose (i, j)-th entry is <X[i], X2[j]>_{L}
        """
        if X2 is None:
            X2 = X
        S = X[:,1:].dot(X2[:,1:].T) - np.outer(X[:,0], X2[:,0])
        # S[S>=-1.] = -1. #- 1e-05
        return S
    
    def distance(self, X, X2=None):
        """
        distance in Hyperboloid
        """
        S = self.Ldot(X, X2)
        S[S>=-1.] = -1. #- 1e-5
        # D = np.arccosh(-1. * S)
        return np.arccosh(-1. * S)

    def parallel_transport(self, V, mu, nu): # TODO debug
        """
        Entry-wise parallel transport of V from nu to mu
        V:(N, Q), mu:(N, Q), nu:(N, Q)
        """
        gamma = (-1.) * self.Ldot_diag(nu, mu)
        tmp = self.Ldot_diag(mu-gamma[:,None]*nu, V) / (gamma + 1.)

        return V + tmp[:,None] * (nu + mu)