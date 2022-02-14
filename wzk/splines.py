import numpy as np


class NURBS:
    def __init__(self, p, degree=3, k=None, w=None):
        self.p = np.array(p)
        self.n_points, self.n_dim = self.p.shape
        self.degree = degree

        self.k = None
        self.set_knotvector(k=k)

        if w is None:
            self.w = np.ones(len(self.p))
        else:
            self.w = np.atleast_1d(w)

        assert len(self.k) == self.degree + self.n_points + 1  # noqa
        assert len(self.p) == len(self.w)

    def __repr__(self):
        return f"NURBS (degree={self.degree}, #points={self.degree})"

    def set_knotvector(self, k):
        if k is None:
            deg2 = int(np.ceil((self.degree+1)/2))
            k = ([0]*deg2 +
                 list(range(self.n_points)) +
                 [self.n_points-1] * (self.degree+1-deg2))

        k = np.atleast_1d(k)
        assert len(k) == self.degree + self.n_points + 1, f"len(k)[{len(k)}] == self.degree[{self.degree}] + self.n_points[{self.n_points}] + 1"
        self.k = k
        self.normalize_knotvector_01()

    def normalize_knotvector_01(self):
        self.k = self.k / self.k.max()

    @staticmethod
    def divide(n, d):
        return np.divide(n, d, out=np.zeros_like(n), where=d != 0)

    def f_in(self, u, i, n):
        f_in = self.divide(u - self.k[i],  self.k[i+n] - self.k[i])
        return f_in

    def n_in(self, u, i, n):
        if n == 0:
            k0 = self.k[i]
            k1 = self.k[i+1]
            n_in = np.logical_and(k0 <= u, u < k1).astype(int)

        else:
            n_in = (self.f_in(u=u, i=i, n=n) * self.n_in(u=u, i=i, n=n-1) +
                    (1-self.f_in(u=u, i=i+1, n=n)) * self.n_in(u=u, i=i+1, n=n-1))
        return n_in

    def evaluate(self, u):
        u = np.atleast_1d(u)
        x = 0.0
        for i in range(self.n_points):
            x = x + self.r_in(u=u, i=i)[:, np.newaxis] * self.p[i:i+1]

        # otherwise, they are zero, weighting is little of somewhere, handle edge cases
        x[u == 0] = self.p[0]
        x[u == 1] = self.p[-1]
        return x

    def evaluate_jac(self, u):
        u = np.atleast_1d(u)
        jac = np.zeros(u.shape + (self.n_points,))
        for i in range(self.n_points):
            jac[..., i] = self.r_in(u=u, i=i)

        return jac

    def r_in(self, u, i):
        # rational basis function
        r_in = self.n_in(u=u, i=i, n=self.degree) * self.w[i]

        temp = 0.0
        for j in range(self.n_points):
            temp += self.n_in(u=u, i=j, n=self.degree) * self.w[j]

        r_in = self.divide(r_in, temp)
        return r_in
