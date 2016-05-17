import numpy as np
from scipy import optimize
from scipy import sparse
import itertools

def genleastsq(y, f, x0, args=(), A=None, b=None, Jf=None, W=1, regl=[], regL=[], regW=[]):
    '''min_x .5*||y-f(x, *args)||^2_W + sum(.5*||regl-regL.dot(x)||^2_regW)
    s.t. A.dot(x) == b
    
    Solve the above problem using optimize.fmin_bfgs on the Lagrangian.
    
    Jf is the Jacobian of f.
    regW should be a list of either scalars or 1d np.ndarrays of weights representing the diagonal
    of the weighting matrices.
    W is also a scalar or a 1d np.ndarray representing the main diagonal
    '''
    # Combine Regulator Matrices
    lws = [w*l for l, w in itertools.izip(regl, regW)]
    assert len(y) == len(f(x0, *args))
    if A is not None and b is not None:
        assert A.shape[0] == b.shape[0] and A.shape[1] == len(x0)
        neq = len(b)
        def grad(x, *args):
            X = x[:-neq]
            g = x[-neq:]
            F = f(X, *args)
            JF = Jf(X, *args)
            Lxs = [L.dot(X) for L in regL]
            out = JF.T.dot(W*(F-y))
            out += sum([lw.dot(Lx-l) for lw, Lx, l in itertools.izip(lws, Lxs, regl)])
            out -= A.T.dot(g)
            out = np.concatenate((out, -A.dot(X)))
            return out
        
        x0 = np.concatenate((x0, np.zeros(neq)))
        return optimize.fsolve(grad, x0, args)[:-neq]
    else:
        def grad(x, *args):
            F = f(x, *args)
            JF = Jf(x, *args)
            Lxs = [L.dot(x) for L in regL]
            out = JF.T.dot(W*(F-y))
            out += sum([lw.dot(Lx-l) for lw, Lx, l in itertools.izip(lws, Lxs, regl)])
            return out
        
        return optimize.fsolve(grad, x0, args)

def mat_sqrt(A):
    w, v = np.linalg.eig(A)
    w = np.where(w < 1e-9, np.zeros(len(w)), w)
    return np.diag(np.sqrt(w)).dot(v.T)

def piecewise_cubic_c2_L2_regulator(Xk):
    '''Matrix L such that ||L.dot(x)||^2 = int_x0^xn s''(u)^2 du'''
    n = len(Xk)
    L = np.zeros((2*n, 2*n))
    for i in range(n-1):
        L[n+i][n+i] += 1./3
        L[n+i+1][n+i+1] += 1./3
        L[n+i][n+i+1] += 1./6
        L[n+i+1][n+i] += 1./6
    
    return L

def piecewise_cubic_c2_L1_regulator(Xk):
    '''Matrix L such that ||L.dot(x)||^2 = int_x0^xn s'(u)^2 du'''
    raise Exception('Not implemented')
    n = len(Xk)
    L = np.zeros((2*n, 2*n))
    for i in range(n-1):
        pass
    
    return L

def tension_penalty_matrix(Xk, sig):
    '''Matrix L such that ||L.dot(x)||^2 = int_x0^xn s''(u)^2 du with tension parameter sig'''
    sig2 = sig**2
    n = len(Xk)
    n2 = 2*n
    A = np.zeros((n2, n2))
    for j in range(n-1):
        dx = Xk[j+1] - Xk[j]
        A[j][j] -= sig2 / dx
        A[j][j+n] += 1. / dx
        A[j][j+1] += sig2 / dx
        A[j][j+n+1] -= 1. / dx
        A[j+1][j] += sig2 / dx
        A[j+1][j+n] -= 1. / dx
        A[j+1][j+1] -= sig2 / dx
        A[j+1][j+1+n] += 1. / dx
    return -.5*(A+A.T)

def tension_spline_matrix(Xk, X, sig):
    '''Given the knots Xk and the evaluation points X and tension parameter sig, returns the matrix S
    such that S.dot(x) is the spline value at each point in X with x the spline parameters'''
    nknots = len(Xk)
    INTERP = []
    for x in X:
        row = nknots*2*[0]
        try:
            n = np.nonzero(x>Xk)[0][-1]
        except:
            row[0] = 1
            INTERP.append(row)
            continue
        if n == nknots - 1:
            row[-1] = 1
        else:
            dx = Xk[n+1] - Xk[n]
            s = np.sinh(sig*dx)
            dxu = Xk[n+1] - x
            dxl = x - Xk[n]
            row[n+1] = dxl / dx
            row[n] = dxu / dx
            row[n+1+nknots] = (np.sinh(sig*dxl)/s-dxl/dx) / sig**2
            row[n+nknots] = (np.sinh(sig*dxu)/s-dxu/dx) / sig**2
        INTERP.append(row)
    return np.array(INTERP)

def tension_spline_1stDer_matrix(Xk, sig):
    '''Matrix L such that ||L.dot(x)||^2 = int_x0^xn s'(u)^2 du with tension parameter sig'''
    nknots = len(Xk)
    A = []
    for n in range(1,nknots-1):
        h0 = Xk[n]-Xk[n-1]
        h1 = Xk[n+1]-Xk[n]
        row = 2*nknots*[0]
        s0 = np.sinh(sig*h0)
        s1 = np.sinh(sig*h1)
        row[n-1] = -sig**2/h0
        row[n]   = sig**2*(1./h1 + 1./h0)
        row[n+1] = -sig**2/h1
        row[n-1+nknots] = 1./h0 - sig/s0
        row[n+nknots]   = sig*np.cosh(sig*h0)/s0-1./h0 + sig*np.cosh(sig*h1)/s1-1./h1
        row[n+1+nknots] = 1./h1 - sig/s1
        A.append(row)
    return np.array(A)

def piecewise_linear_matrix(Xk, X):
    '''Given knot points Xk and evaluation points X, returns a matrix S such that S.dot(x) returns the
    interpolated value of the spline at each point in X for the spline parameters x'''
    nknots = len(Xk)
    INTERP = []
    for x in X:
        try:
            n = np.nonzero(x>Xk)[0][-1]
        except:
            row = nknots*[0]
            row[0] = 1
            INTERP.append(row)
            continue
        if n == nknots - 1:
            row = nknots*[0]
            row[-1] = 1
        else:
            row = nknots*[0]
            dx = Xk[n+1] - Xk[n]
            row[n+1] = (x - Xk[n]) / dx
            row[n] = (Xk[n+1] - x) / dx
        INTERP.append(row)
    return np.array(INTERP)

def piecewise_cubic_matrix(Xk, X):
    '''Given knot points Xk and evaluation points X, returns a matrix S such that S.dot(x) returns the
    interpolated value of the spline at each point in X for the spline parameters x'''
    nknots = len(Xk)
    INTERP = []
    for x in X:
        try:
            n = np.nonzero(x>Xk)[0][-1]
        except:
            row = 2*nknots*[0]
            row[0] = 1
            INTERP.append(row)
            continue
        if n == nknots - 1:
            row = 2*nknots*[0]
            row[nknots-1] = 1
        else:
            row = 2*nknots*[0]
            dx = Xk[n+1] - Xk[n]
            t = (x - Xk[n])/dx
            T = 1.-t
            row[n]   = T + t*T*T - t*t*T
            row[n+1] = t - t*T*T + t*t*T
            row[n+nknots]   = t*T*T*dx
            row[n+1+nknots] = -t*t*T*dx
        INTERP.append(row)
    return np.array(INTERP)

def piecewise_cubic_c2_matrix(Xk, X):
    '''Given knot points Xk and evaluation points X, returns a matrix S such that S.dot(x) returns the
    interpolated value of the spline at each point in X for the spline parameters x. Must be paired 
    with the constraint A.dot(x) == b with A and b coming from piecewise_cubic_c2_cons_matrix(Xk).'''
    nknots = len(Xk)
    INTERP = []
    for x in X:
        try:
            n = np.nonzero(x>Xk)[0][-1]
        except:
            row = 2*nknots*[0]
            row[0] = 1
            INTERP.append(row)
            continue
        if n == nknots - 1:
            row = 2*nknots*[0]
            row[nknots-1] = 1
        else:
            row = 2*nknots*[0]
            dx = Xk[n+1] - Xk[n]
            row[n]   = (Xk[n+1]-x)/dx
            row[n+1] = (x - Xk[n])/dx
            row[n+nknots]   = ((Xk[n+1]-x)**3/dx-(Xk[n+1]-x)*dx)/6.
            row[n+1+nknots] = ((x-Xk[n])**3/dx-(x - Xk[n])*dx)/6.
        INTERP.append(row)
    return np.array(INTERP)

def piecewise_cubic_c2_cons_matrix(Xk):
    '''returns a matrix A and vector b such that A.dot(x) == b for spline parameter vector x.'''
    nknots = len(Xk)
    A = []
    b = []
    for i in range(1, nknots-1):
        row = nknots*2*[0.]
        dxl = Xk[i] - Xk[i-1]
        dxu = Xk[i+1] - Xk[i]
        row[i+1] = -1./dxu
        row[i] = 1./dxu + 1./dxl
        row[i-1] = -1./dxl
        row[nknots+i+1] = dxu/6.
        row[nknots+i] = (dxu+dxl)/3.
        row[nknots+i-1] = dxl/6.
        A.append(row)
        b.append(0)
    return np.array(A), np.array(b)

class LinearSpline(object):
    def __init__(self, Yk, Xk, X, C=None, W=None, A=None, b=None, L1=None):
        """
        min_x ||Yk - C.dot(S.dot(x))||^2_W + ||L.dot(x)||^2_L1
        s.t A.dot(x) = b
        
        where:
            Yk - vector, nsamples x 1
                target values
            Xk - vector, nknots x 1
                knot values
            C - matrix, nsamples x nknots
                transforms the spline estimates into the target space
            W - PD diagonal matrix, nsamples x nsamples
                weighs the importance of each point
            A - matrix, ncons x nknots
                extra interpolation constraints
            b - vector, ncons x 1
                extra interpolation constraint bounds
            L1 - PD diagonal matrix, nsamples x nsamples
                the weight of the 1st derivative regulator. If None, not used. If scalar, L1*I.
                This replicates integral_Xk[0]^Xk[-1] (l'(x))^2 dx.
        """
        self.Yk = Yk
        self.Xk = Xk
        self.X = X
        self.nsamples = len(X)
#         assert len(self.Yk) == self.nsamples, 'Yk must have the same length as X'
        self.nknots = len(Xk)
        self.nvars = self.nknots
        self._isfit = False
        self.C = C if C is not None else np.eye(self.nsamples)
        self.W = W if W is not None else np.eye(self.C.shape[0])
        self.A = A
        self.b = b
        self.L1 = L1*np.eye(self.nknots) if L1 is not None and np.array(L1).shape==() else L1
        
        self.s = piecewise_linear_matrix(self.Xk, self.X)
        nvars = self.s.shape[1]
        
        if self.L1 is not None:
            L = np.zeros((self.nknots, self.nknots))
            for i in range(self.nknots-1):
                dx = self.Xk[i+1] - self.Xk[i]
                L[i][i] += 1./dx
                L[i+1][i+1] += 1./dx
                L[i][i+1] -= 1./dx
                L[i+1][i] -= 1./dx
            w, v = np.linalg.eig(L)
            w = np.where(w < 1e-9, np.zeros(len(w)), w)
            self.L = np.diag(np.sqrt(w)).dot(v.T)
    
        wgt_diags = [self.W[i][i]/float(self.nsamples) for i in range(self.W.shape[0])] + \
                    ([self.L1[i][i]/float(self.nvars) for i in range(self.nknots)] if self.L1 is not None else [])
    
        self.wgt = sparse.diags(wgt_diags)
        self._setYandS()
        self.updateMat = None
    
    def _setYandS(self):
        if self.L1 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nknots)))
            self.S = np.concatenate((self.C.dot(self.s), self.L))
        else:
            self.Y = self.Yk
            self.S = self.C.dot(self.s)
    
    def _setY(self):
        if self.L1 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nknots)))
        else:
            self.Y = self.Yk
    
    def fit(self, Yk=None):
        """Fit the spline parameters self.x"""
        if Yk is not None:
            self._setYandS()
        
        self.sw  = np.array(self.S.T.dot(self.wgt.todense()))
        self.swy = self.sw.dot(self.Y)
        self.sws = self.sw.dot(self.S)
        self.swssw = np.linalg.solve(self.sws, self.sw)
        
        if self.A is not None:
            self.aswsa = self.A.dot(np.linalg.solve(self.sws, self.A.T))
            r2 = np.linalg.solve(self.aswsa, self.b - self.A.dot(self.swssw.dot(self.Y)))
            r = self.swy + self.A.T.dot(r2)
            self.x = np.linalg.solve(self.sws, r)
        else:
            self.x = np.linalg.solve(self.sws, self.swy)
        
        self._isfit = True
        return self.x
    
    def update(self, Yk):
        """Update the spline parameters for the new target vector Yk using jacobian update instead
        of refitting"""
        assert self._isfit, 'Must fit spline before updating'
        oldY = self.Y
        self.Yk = Yk
        self._setY()
        if self.A is not None:
            if not self.updateMat:
                r2 = self.A.dot(self.swssw)
                r = np.linalg.solve(self.aswsa, r2)
                self.updateMat = self.swssw - self.A.T.dot(r)
            self.x += self.updateMat.dot(self.Y-oldY)
        else:
            self.x += self.swssw.dot(self.Y-oldY)
        return self.x
    
    def x_estimates(self):
        assert self._isfit, 'Must fit spline before estimating'
        return self.s.dot(self.x)
    
    def cost_estimates(self):
        assert self._isfit, 'Must fit spline before estimating'
        return self.C.dot(self.s.dot(self.x))
    
    def est(self, x):
        """Return the interpolated value at point x"""
        assert self._isfit, 'Must fit spline before estimating'
        try:
            len(x)
        except:
            x = np.array([x])
        return piecewise_linear_matrix(self.Xk, np.array(x)).dot(self.x)

class CubicSpline(object):
    def __init__(self, Yk, Xk, X, C=None, W=None, A=None, b=None, L1=None, L2=None, natural=False, notaknot=False, c2=True):
        """
        min_x ||Yk - C.dot(S.dot(x))||^2_W + ||L.dot(x)||^2_L1 + ||R.dot(x)||^2_L2
        s.t A.dot(x) = b
        
        where:
            Yk - vector, nsamples x 1
                target values
            Xk - vector, nknots x 1
                knot values
            X - vector, nsamples x 1
                the points where the spline is evaluated in the objective. Used to construct the S matrix.
            C - matrix, nsamples x nknots
                transforms the spline estimates into the target space
            W - PD diagonal matrix, nsamples x nsamples
                weighs the importance of each point
            A - matrix, ncons x nknots
                extra interpolation constraints
            b - vector, ncons x 1
                extra interpolation constraint bounds
            L1 - PD diagonal matrix, nsamples x nsamples
                the weight of the 1st derivative regulator. If None, not used. If scalar, L1*I.
                Replicates integral_x0^xn q'(x)^2 dx
            L2 - PD diagonal matrix, nsamples x nsamples
                the weight of the 2nd derivative regulator. If None, not used. If scalar, L1*I.
                Replicates integral_x0^xn q''(x)^2 dx
            c2 - bool, default True
                If True, force the cubic spline to be C2 (twice continuously differentiable).
        
        End Point Conditions: (mutually exclusive)
            natural - bool
                the 2nd derivative at the end points is zero
            notaknot - bool
                the 2nd derivatives at the 1st outer points must match
        """
        assert not (natural and notaknot), 'can choose at most one endpoint condition'
        self._c2 = c2
        self.Yk = Yk
        self.Xk = Xk
        self.X = X
        self.nsamples = len(X)
#         assert len(self.Yk) == self.nsamples, 'Yk must have the same length as X'
        self.nknots = len(Xk)
        self.nvars = 2*self.nknots
        self._isfit = False
        self.C = C if C is not None else np.eye(self.nsamples)
        self.W = W if W is not None else np.eye(self.C.shape[0])
        self.A = np.array(A) if A is not None else A
        self.b = np.array(b) if b is not None else b
        self.L1 = L1*np.eye(self.nvars) if L1 is not None and np.array(L1).shape==() else L1
        self.L2 = L2*np.eye(self.nvars) if L2 is not None and np.array(L2).shape==() else L2
        
        if c2:
            A, b = piecewise_cubic_c2_cons_matrix(self.Xk)
            if self.A is not None:
                self.A = np.concatenate((self.A, A))
                self.b = np.concatenate((self.b, b))
            else:
                self.A, self.b = A, b
            self.s = piecewise_cubic_c2_matrix(self.Xk, self.X)
        else:
            self.s = piecewise_cubic_matrix(self.Xk, self.X)
        
        if self.L2 is not None:
            R = np.zeros((2*self.nknots, 2*self.nknots))
            for i in range(self.nknots-1):
                dx = self.Xk[i+1] - self.Xk[i]
                R[i][i] += 12./dx**3
                R[i+1][i+1] += 12./dx**3
                R[i][i+1] -= 2*12./dx**3
                R[self.nknots+i][self.nknots+i] += 4./dx
                R[self.nknots+i+1][self.nknots+i+1] += 4./dx
                R[self.nknots+i][self.nknots+i+1] += 4./dx
                R[i][self.nknots+i] -= 4./dx**2
                R[i][self.nknots+i+1] -= 4./dx**2
                R[i+1][self.nknots+i] += 4./dx**2
                R[i+1][self.nknots+i+1] += 4./dx**2
            R = .5*(R + R.T)
            w, v = np.linalg.eig(R)
            w = np.where(w < 1e-9, np.zeros(len(w)), w)
            self.R = np.diag(np.sqrt(w)).dot(v.T)
            
        if self.L1 is not None:
            L = np.zeros((2*self.nknots, 2*self.nknots))
            for i in range(self.nknots-1):
                dx = self.Xk[i+1] - self.Xk[i]
                L[i][i] += 1.2/dx
                L[i+1][i+1] += 1.2/dx
                L[i][i+1] -= 2*1.2/dx
                L[self.nknots+i][self.nknots+i] += 2.*dx/15.
                L[self.nknots+i+1][self.nknots+i+1] += 2.*dx/15.
                L[self.nknots+i][self.nknots+i+1] -= dx/15.
                L[i][self.nknots+i] -= .2
                L[i][self.nknots+i+1] -= .2
                L[i+1][self.nknots+i] += .2
                L[i+1][self.nknots+i+1] += .2
            L = .5*(L + L.T)
            w, v = np.linalg.eig(L)
            w = np.where(w < 1e-9, np.zeros(len(w)), w)
            self.L = np.diag(np.sqrt(w)).dot(v.T)
                
        if c2 and (natural or notaknot):
            A = []
            b = []
            
            if natural:
                row = self.nvars*[0]
                dx = Xk[1] - Xk[0]
                row[0] = -3./dx**2
                row[1] =  3./dx**2
                row[self.nknots]   = -2./dx
                row[1+self.nknots] = -1./dx
                A.append(row)
                b.append(0)
                
                row = self.nvars*[0]
                dx = Xk[-1] - Xk[-2]
                row[self.nknots-1] = -3./dx**2
                row[self.nknots-2] =  3./dx**2
                row[-1] = 2./dx
                row[-2] = 1./dx
                A.append(row)
                b.append(0)
            elif notaknot:
                for n in (0, self.nknots-3):
                    row = self.nvars*[0]
                    dx1 = Xk[n+1] - Xk[n]
                    dx2 = Xk[n+2] - Xk[n+1]
                    row[n]   =  2./dx1**3
                    row[n+1] = -2.*(1./dx1**3+1./dx2**3)
                    row[n+2] =  2./dx2**3
                    row[self.nknots+n]   =  1./dx1**2
                    row[self.nknots+n+1] =  1./dx1**2 - 1./dx2**2
                    row[self.nknots+n+2] = -1./dx2**2
                    A.append(row)
                    b.append(0)
            if self.A is not None:
                self.A = np.concatenate((self.A, A))
                self.b = np.concatenate((self.b, b))
            else:
                self.A = A
                self.b = b
        if self.A is not None:
            self.A = np.array(A)
            self.b = np.array(b)
    
        wgt_diags = [self.W[i][i]/float(self.nsamples) for i in range(self.W.shape[0])] + \
                    ([self.L1[i][i]/float(self.nvars) for i in range(self.nvars)] if self.L1 is not None else []) + \
                    ([self.L2[i][i]/float(self.nvars) for i in range(self.nvars)] if self.L2 is not None else [])
    
        self.wgt = sparse.diags(wgt_diags)
        self._setYandS()
        self.updateMat = None
    
    def _setYandS(self):
        if self.L1 is not None and self.L2 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(2*self.nvars)))
            self.S = np.concatenate((self.C.dot(self.s), self.L, self.R))
        elif self.L1 is not None and self.L2 is None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nvars)))
            self.S = np.concatenate((self.C.dot(self.s), self.L))
        elif self.L1 is None and self.L2 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nvars)))
            self.S = np.concatenate((self.C.dot(self.s), self.R))
        else:
            self.Y = self.Yk
            self.S = self.C.dot(self.s)
    
    def _setY(self):
        if self.L1 is not None and self.L2 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nvars*2)))
        elif self.L1 is not None and self.L2 is None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nvars)))
        elif self.L1 is None and self.L2 is not None:
            self.Y = np.concatenate((self.Yk, np.zeros(self.nvars)))
        else:
            self.Y = self.Yk
    
    def fit(self, Yk=None):
        """Fit the spline parameters self.x"""
        if Yk is not None:
            self.Yk = Yk
            self._setYandS()
        
        self.sw  = np.array(self.S.T.dot(self.wgt.todense()))
        self.swy = self.sw.dot(self.Y)
        self.sws = self.sw.dot(self.S)
        self.swssw = np.linalg.solve(self.sws, self.sw)
        
        if self.A is not None:
            self.aswsa = self.A.dot(np.linalg.solve(self.sws, self.A.T))
            r2 = np.linalg.solve(self.aswsa, self.b - self.A.dot(self.swssw.dot(self.Y)))
            r = self.swy + self.A.T.dot(r2)
            self.x = np.linalg.solve(self.sws, r)
        else:
            self.x = np.linalg.solve(self.sws, self.swy)
        
        self._isfit = True
        return self.x
    
    def update(self, Yk):
        """Update the spline parameters for the new target vector Yk using jacobian update instead
        of refitting"""
        assert self._isfit, 'Must fit spline before updating'
        oldY = self.Y
        self.Yk = Yk
        self._setY()
        if self.A is not None:
            if self.updateMat is None:
                r2 = self.A.dot(self.swssw)
                r = np.linalg.solve(self.aswsa, r2)
                self.updateMat = self.swssw - self.A.T.dot(r)
            self.x += self.updateMat.dot(self.Y-oldY)
        else:
            self.x += self.swssw.dot(self.Y-oldY)
        return self.x
    
    def x_estimates(self):
        assert self._isfit, 'Must fit spline before estimating'
        return self.s.dot(self.x)
    
    def cost_estimates(self):
        assert self._isfit, 'Must fit spline before estimating'
        return self.C.dot(self.s.dot(self.x))
    
    def est(self, x):
        """Return the interpolated value at point x"""
        assert self._isfit, 'Must fit spline before estimating'
        try:
            len(x)
        except:
            x = np.array([x])
        if self._c2:
            return piecewise_cubic_c2_matrix(self.Xk, np.array(x)).dot(self.x)
        else:
            return piecewise_cubic_matrix(self.Xk, np.array(x)).dot(self.x)

