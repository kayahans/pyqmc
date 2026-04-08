import numpy as np
import pyqmc.gpu as gpu

""" 
Collection of 3d function objects. Each has a dictionary parameters, which corresponds
to any variational parameters the funtion has.

"""


class GaussianFunction:
    r"""A representation of a Gaussian:
    :math:`f(r) = \exp(-\alpha r^2)`

    where :math:`\alpha` can be accessed through parameters['exponent']
    """

    def __init__(self, exponent):
        self.parameters = {"exponent": exponent}

    def value(self, x, r):
        return np.exp(-self.parameters["exponent"] * r * r)

    def gradient(self, x, r):
        v = self.value(x, r)
        return -2 * self.parameters["exponent"] * x * v[..., np.newaxis]

    def gradient_value(self, x, r):
        v = self.value(x, r)
        g = -2 * self.parameters["exponent"] * x * v[..., np.newaxis]
        return g, v

    def laplacian(self, x, r):
        v = self.value(x, r)
        alpha = self.parameters["exponent"]
        return (4 * alpha * alpha * x * x - 2 * alpha) * v[..., np.newaxis]

    def gradient_laplacian(self, x, r):
        v = self.value(x, r)[..., np.newaxis]
        alpha = self.parameters["exponent"]
        grad = -2 * alpha * x * v
        lap = (4 * alpha * alpha * x * x - 2 * alpha) * v
        return grad, lap

    def pgradient(self, x, r):
        r"""Returns parameters gradient.

        :parameter x: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: parameter gradient {'exponent': :math:`\frac{\partial f}{\partial \alpha}`}
        :rtype: dictionary
        """
        r2 = r * r
        return {"exponent": -r2 * np.exp(-self.parameters["exponent"] * r2)}


class PadeFunction:
    r"""
    a_k(r) = (alpha_k*r/(1+alpha_k*r))^2
    alpha_k = alpha/2^k, k starting at 0

    :math:`a_k(r) = \left( \frac{\alpha_k r}{1 + \alpha_k r} \right)^2`
    where
    :math:`\alpha_k = \frac{\alpha}{2^k}`, :math:`k` starting at 0
    """

    def __init__(self, alphak):
        self.parameters = {"alphak": alphak}

    def value(self, rvec, r):
        a = self.parameters["alphak"] * r
        return (a / (1 + a)) ** 2

    def gradient(self, rvec, r):
        a = self.parameters["alphak"] * r[..., np.newaxis]
        return 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3 * rvec

    def gradient_value(self, rvec, r):
        a = self.parameters["alphak"] * r
        value = (a / (1 + a)) ** 2
        a = a[..., np.newaxis]
        grad = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3 * rvec
        return grad, value

    def laplacian(self, rvec, r):
        a = self.parameters["alphak"] * r[..., np.newaxis]
        # lap = 6*self.parameters['alphak']**2 * (1+a)**(-4) #scalar formula
        lap = (
            2
            * self.parameters["alphak"] ** 2
            * (1 + a) ** (-3)
            * (1 - 3 * a / (1 + a) * (rvec / r[..., np.newaxis]) ** 2)
        )
        return lap

    def gradient_laplacian(self, rvec, r):
        a = self.parameters["alphak"] * r[..., np.newaxis]
        temp = 2 * self.parameters["alphak"] ** 2 / (1 + a) ** 3
        grad = temp * rvec
        lap = temp * (1 - 3 * a / (1 + a) * (rvec / r[..., np.newaxis]) ** 2)
        return grad, lap

    def pgradient(self, rvec, r):
        r"""Returns gradient with respect to parameter alphak

        :parameter rvec: (nconfig,...,3)
        :parameter r: (nconfig,...)
        :returns: parameter gradient  {'alphak': :math:`\frac{\partial a_k}{\partial \alpha_k}`}
        :rtype: dictionary
        """
        a = self.parameters["alphak"] * r
        akderiv = 2 * a / (1 + a) ** 3 * r
        return {"alphak": akderiv}


@gpu.fuse()
def polypadevalue(z, beta):
    z2 = z * z
    p = z2 * (6 - 8 * z + 3 * z2)
    return (1 - p) / (1 + beta * p)


@gpu.fuse()
def polypadegradvalue(r, beta, rcut):
    z = r / rcut
    z1 = z - 1
    z12 = z1 * z1
    p = (3 * z12 + 4 * z1) * z12 + 1
    obp = 1 / (1 + beta * p)
    dpdz = 12 * z * (z * z - 2 * z + 1)
    dbdp = -(1 + beta) * obp * obp
    dzdx_rvec = 1 / (r * rcut)
    grad_rvec = dbdp * dpdz * dzdx_rvec
    value = (1 - p) * obp
    return grad_rvec, value


class PolyPadeFunction:
    r"""

    .. math:: b(r) = \frac{1-p(z)}{1+\beta p(z)}, \quad z = r/r_{\rm cut} 

    where :math:`p(z) = 6z^2 - 8z^3 + 3z^4`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, beta, rcut):
        self.parameters = {
            "beta": gpu.cp.asarray(beta),
            "rcut": gpu.cp.asarray(rcut),
        }

    def value(self, rvec, r):
        mask = r < self.parameters["rcut"]
        z = r / self.parameters["rcut"]
        func = polypadevalue(z, self.parameters["beta"])
        return gpu.cp.where(mask, func, 0.0)

    def gradient_value(self, rvec, r):
        mask = r < self.parameters["rcut"]
        grad_rvec, val = polypadegradvalue(
            r,
            self.parameters["beta"],
            self.parameters["rcut"],
        )
        grad_full = rvec * grad_rvec[..., np.newaxis]
        grad = gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)
        value = gpu.cp.where(mask, val, 0.0)
        return grad, value

    def gradient(self, rvec, r):
        mask = r < self.parameters["rcut"]
        rr = r[..., np.newaxis]
        rcut = self.parameters["rcut"]
        z = rr / rcut
        p = z * z * (6 - 8 * z + 3 * z * z)
        dpdz = 12 * z * (z * z - 2 * z + 1)
        dbdp = -(1 + self.parameters["beta"]) / (1 + self.parameters["beta"] * p) ** 2
        dzdx = rvec / (rr * rcut)
        grad_full = dbdp * dpdz * dzdx
        return gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)

    def laplacian(self, rvec, r):
        return self.gradient_laplacian(rvec, r)[1]

    def gradient_laplacian(self, rvec, r):
        mask = r < self.parameters["rcut"]
        rr = r[..., np.newaxis]
        rcut = self.parameters["rcut"]
        z = rr / rcut
        z1 = z - 1
        z12 = z1 * z1
        beta = self.parameters["beta"]

        p = (3 * z12 + 4 * z1) * z12 + 1
        obp = 1 / (1 + beta * p)
        dpdz = 12 * z * z12
        dbdp = -(1 + beta) * obp * obp
        dzdx = rvec / (rr * rcut)
        grad_full = dbdp * dpdz * dzdx
        d2pdz2_over_dpdz = (3 * z - 1) / (z * z1)
        d2bdp2_over_dbdp = -2 * beta * obp
        d2zdx2 = (1 - (rvec / rr) ** 2) / (rr * rcut)
        lap_full = dbdp * dpdz * d2zdx2 + (
            grad_full * (d2bdp2_over_dbdp * dpdz * dzdx + d2pdz2_over_dpdz * dzdx)
        )
        grad = gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)
        lap = gpu.cp.where(mask[..., np.newaxis], lap_full, 0.0)
        return grad, lap

    def pgradient(self, rvec, r):
        r"""
        :returns: parameter gradient {'rcut': :math:`\frac{\partial b}{\partial r_{\rm cut}}`, 'beta': :math:`\frac{\partial b}{\partial \beta}`}
        :rtype: dictionary
        """
        mask = r >= self.parameters["rcut"]
        z = r / self.parameters["rcut"]
        z1 = z - 1
        z12 = z1 * z1
        beta = self.parameters["beta"]

        p = (3 * z12 + 4 * z1) * z12 + 1
        obp = 1 / (1 + beta * p)
        dpdz = 12 * z * z12
        dbdp = -(1 + beta) * obp * obp
        derivrcut = dbdp * dpdz * (-z / self.parameters["rcut"])
        derivbeta = -p * (1 - p) * obp * obp
        pderiv = {
            "rcut": gpu.cp.where(mask, 0.0, derivrcut),
            "beta": gpu.cp.where(mask, 0.0, derivbeta),
        }
        return pderiv


class CutoffCuspFunction:
    r"""
    .. math:: b(r) = -\frac{p(r/r_{\rm cut})}{1+\gamma*p(r/r_{\rm cut})} + \frac{1}{3+\gamma} 

    where
    :math:`p(y) = y - y^2 + y^3/3`

    This function is positive at small r, and is zero for :math:`r \ge r_{\rm cut}`.
    """

    def __init__(self, gamma, rcut):
        self.parameters = {"gamma": gamma, "rcut": rcut}

    def value(self, rvec, r):
        mask = r < self.parameters["rcut"]
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        y = r / rcut
        y1 = y - 1
        p = (y1 * y1 * y1 + 1) / 3
        inner = -p / (1 + gamma * p) + 1 / (3 + gamma)
        func = gpu.cp.where(mask, inner, 0.0)
        return func * rcut

    def gradient(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        rr = r[..., np.newaxis]
        y = rr / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * rr)

        grad_full = -rvec * a * c * rcut
        return gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)

    def gradient_value(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        rr = r[..., np.newaxis]
        y = rr / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        ogb = 1 / (1 + gamma * b)
        c = ogb * ogb / (rcut * rr)

        grad_full = -rvec * a * c * rcut
        grad = gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)
        value_inner = -(b * ogb)[..., 0] + 1 / (3 + gamma)
        value = gpu.cp.where(mask, value_inner, 0.0)
        return grad, value * rcut

    def laplacian(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        rr = r[..., np.newaxis]
        y = rr / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * rr)

        temp = 2 * y1 / (rcut * rr)
        temp -= a / rr**2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap_full = -rcut * c * (a + rvec**2 * temp)
        return gpu.cp.where(mask[..., np.newaxis], lap_full, 0.0)

    def gradient_laplacian(self, rvec, r):
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r < rcut
        rr = r[..., np.newaxis]
        y = rr / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        c = 1 / (1 + gamma * b) ** 2 / (rcut * rr)

        grad_full = -rcut * a * c * rvec
        temp = 2 * y1 / (rcut * rr)
        temp -= a / rr**2
        temp -= 2 * a * a * c * gamma * (1 + gamma * b)
        lap_full = -rcut * c * (a + rvec**2 * temp)
        grad = gpu.cp.where(mask[..., np.newaxis], grad_full, 0.0)
        lap = gpu.cp.where(mask[..., np.newaxis], lap_full, 0.0)
        return grad, lap

    def pgradient(self, rvec, r):
        r"""

        :parameter rvec: (nconf,...,3)
        :parameter r: (nconfig,...)
        :returns: parameter derivatives {'rcut': :math:`\frac{\partial b}{\partial r_{\rm cut}}`, 'gamma': :math:`\frac{\partial b}{\partial \gamma}`}
        :rtype: dict
        """
        rcut = self.parameters["rcut"]
        gamma = self.parameters["gamma"]
        mask = r > rcut
        y = r / rcut
        y1 = y - 1

        a = y1 * y1
        b = (a * y1 + 1) / 3
        ogb = 1 / (1 + gamma * b)
        c = a * ogb * ogb / (rcut * r)
        val = -b * ogb + 1 / (3 + gamma)

        dfdrcut = y * a * ogb * ogb + val
        dfdgamma = ((b * ogb) ** 2 - 1 / (3 + gamma) ** 2) * rcut
        func = {
            "rcut": gpu.cp.where(mask, 0.0, dfdrcut),
            "gamma": gpu.cp.where(mask, 0.0, dfdgamma),
        }

        return func


def test_func3d_gradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    numeric = gpu.cp.zeros(rvec.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        plusval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        pos[..., d] -= 2 * delta
        minuval = bf.value(pos, np.linalg.norm(pos, axis=-1))
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(grad - numeric))
    return gpu.asnumpy(maxerror)


def test_func3d_laplacian(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 5, 10, 3))  # Internal indices irrelevant
    r = np.linalg.norm(rvec, axis=-1)
    lap = bf.laplacian(rvec, r)
    numeric = gpu.cp.zeros(rvec.shape)
    for d in range(3):
        pos = rvec.copy()
        pos[..., d] += delta
        r = np.linalg.norm(pos, axis=-1)
        plusval = bf.gradient(pos, r)[..., d]
        pos[..., d] -= 2 * delta
        r = np.linalg.norm(pos, axis=-1)
        minuval = bf.gradient(pos, r)[..., d]
        numeric[..., d] = (plusval - minuval) / (2 * delta)
    maxerror = np.max(np.abs(lap - numeric))
    return gpu.asnumpy(maxerror)


def test_func3d_gradient_laplacian(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    lap = bf.laplacian(rvec, r)
    andgrad, andlap = bf.gradient_laplacian(rvec, r)
    graderr = np.amax(np.abs(grad - andgrad))
    laperr = np.amax(np.abs(lap - andlap))
    return {"grad": graderr, "lap": laperr}


def test_func3d_gradient_value(bf):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    grad = bf.gradient(rvec, r)
    val = bf.value(rvec, r)
    andgrad, andval = bf.gradient_value(rvec, r)
    graderr = np.linalg.norm((grad - andgrad))
    valerr = np.linalg.norm((val - andval))
    return {"grad": graderr, "val": valerr}


def test_func3d_pgradient(bf, delta=1e-5):
    rvec = gpu.cp.asarray(np.random.randn(150, 10, 3))
    r = np.linalg.norm(rvec, axis=-1)
    pgrad = bf.pgradient(rvec, r)
    numeric = {k: gpu.cp.zeros(v.shape) for k, v in pgrad.items()}
    maxerror = {k: np.zeros(v.shape) for k, v in pgrad.items()}
    for k in pgrad.keys():
        bf.parameters[k] += delta
        plusval = bf.value(rvec, r)
        bf.parameters[k] -= 2 * delta
        minuval = bf.value(rvec, r)
        bf.parameters[k] += delta
        numeric[k] = (plusval - minuval) / (2 * delta)
        maxerror[k] = gpu.asnumpy(np.max(np.abs(pgrad[k] - numeric[k])))
        if maxerror[k] > 1e-5:
            print(k, "\n", pgrad[k] - numeric[k])
    return maxerror
