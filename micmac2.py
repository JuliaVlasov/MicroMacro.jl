import numpy as np
import scipy as sp
from scipy.fftpack import fft, ifft
from numpy import exp, conj, abs, pi, sum, sqrt, arange, newaxis
from numpy import sin, cos, shape
from numpy import linspace, zeros, concatenate, array, ravel
from numpy.linalg import norm
import os, time

def reconstr(U, temps, T, ntau):

    W = arange(ntau, dtype=complex)
    W[ntau // 2:] -= ntau
    W = exp(1j * 2 * pi / T * W * temps)
    V = fft(U, axis=0)
    UA = sum(V * W[:,newaxis],axis=0) / ntau

    return UA

def ftau(m, t, fft_u, fft_v):

    u = ifft(exp(1j * t * m.A1) * fft_u) * m.matr
    v = ifft(exp(1j * t * m.A1) * fft_v) * m.matr

    z = (u + conj(v)) / 2

    fz1 = abs(z) ** (2 * m.sigma) * z

    u = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) 
    v = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) 

    u = u * fft(m.conjmatr * fz1)
    v = v * fft(m.conjmatr * conj(fz1))

    return u, v


def duftau(m, t, fft_u, fft_v, fft_du, fft_dv ):

    sigma = 1

    u = ifft(exp(1j * t * m.A1) * fft_u * m.matr)
    v = ifft(exp(1j * t * m.A1) * fft_v * m.matr)

    du = ifft(exp(1j * t * m.A1) * fft_du * m.matr)
    dv = ifft(exp(1j * t * m.A1) * fft_dv * m.matr)

    z  = (u  + conj(v)) / 2
    dz = (du + conj(dv)) / 2

    fz1 = 2 * abs(z)**2 * dz + z**2 * conj(dz)
    u = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * fft(m.conjmatr * fz1)
    v = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * fft(m.conjmatr * conj(fz1))

    return u, v


def dtftau(m, t, fft_u, fft_v):

    # attention ici je n'ai code' que le cas sigma=1
    sigma = 1

    u  = ifft(exp(1j * t * m.A1) * fft_u * m.matr)
    v  = ifft(exp(1j * t * m.A1) * fft_v * m.matr)
    du = ifft(exp(1j * t * m.A1) * (1j * m.A1) * fft_u) * m.matr
    dv = ifft(exp(1j * t * m.A1) * (1j * m.A1) * fft_v) * m.matr

    z  = ( u + conj(v)) / 2
    dz = (du + conj(dv)) / 2

    fz1 = 2 * abs(z)**2 * dz + z**2 * conj(dz)
    u1 = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * fft(m.conjmatr * fz1)
    v1 = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * fft(m.conjmatr * conj(fz1))

    fz1 = abs(z)**2 * z
    u2 = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * (-1j * m.A1) * fft(m.conjmatr * fz1)
    v2 = -1j * m.llambda * m.A2 * exp(-1j * t * m.A1) * (-1j * m.A1) * fft(m.conjmatr * conj(fz1))

    u = u1 + u2
    v = v1 + v2

    return u, v

def init_2(m, t, fft_u0, fft_v0):

    champu, champv = ftau(m, t, fft_u0, fft_v0)

    champu_fft = fft(champu, axis=0)
    champv_fft = fft(champv, axis=0)

    champu_fft[0, :] = 0 * champu_fft[0, :]
    champv_fft[0, :] = 0 * champv_fft[0, :]

    champu = ifft(champu_fft / (1j * m.ktau), axis=0)
    champv = ifft(champv_fft / (1j * m.ktau), axis=0)

    fft_ubar = fft_u0 - m.epsilon * champu[0, :]
    fft_vbar = fft_v0 - m.epsilon * champv[0, :]

    C1u, C1v = C1(m, t, fft_ubar, fft_vbar)

    fft_ug = fft_u0 - C1u[0, :]
    fft_vg = fft_v0 - C1v[0, :]

    return fft_ubar, fft_vbar, fft_ug, fft_vg

def champs_2(m, t, fft_ubar, fft_vbar, fft_ug, fft_vg):

    champu, champv = ftau(m, t, fft_ubar, fft_vbar)

    champu = fft(champu, axis=0)
    champv = fft(champv, axis=0)

    champu[0, :].fill(0)  
    champv[0, :].fill(0) 

    dtauh1u = ifft(champu, axis=0)
    dtauh1v = ifft(champv, axis=0)

    champu = champu / (1j * m.ktau)
    champv = champv / (1j * m.ktau)

    champu = m.epsilon * ifft(champu, axis=0)
    champv = m.epsilon * ifft(champv, axis=0)

    champu = fft_ubar + champu
    champv = fft_vbar + champv

    ffu, ffv = ftau(m, t, champu + fft_ug, champv + fft_vg)

    champu, champv = ftau(m, t, champu, champv)

    champu = fft(champu, axis=0)
    champv = fft(champv, axis=0)

    champubar = champu[0, :] / m.ntau
    champvbar = champv[0, :] / m.ntau

    champu1, champv1 = duftau(m, t, fft_ubar, fft_vbar, champubar, champvbar)

    champu2, champv2 = dtftau(m, t, fft_ubar, fft_vbar)

    champu = fft(champu1 + champu2, axis=0)
    champv = fft(champv1 + champv2, axis=0)

    champu[0, :].fill(0)  
    champv[0, :].fill(0)  

    champu = champu / (1j * m.ktau)
    champv = champv / (1j * m.ktau)

    champu = m.epsilon * ifft(champu, axis=0)
    champv = m.epsilon * ifft(champv, axis=0)

    champu = champubar + champu
    champv = champvbar + champv

    champu = ffu - dtauh1u - champu
    champv = ffv - dtauh1v - champv

    champu = fft(champu, axis=0)
    champv = fft(champv, axis=0)

    champmoyu = champu[0, :] / m.ntau
    champmoyv = champv[0, :] / m.ntau

    champu[0, :].fill(0)  
    champv[0, :].fill(0)  

    champu = champu / (1j * m.ktau)
    champv = champv / (1j * m.ktau)

    champu = ifft(champu, axis=0)
    champv = ifft(champv, axis=0)

    return champubar, champvbar, champu, champv, champmoyu, champmoyv


def C1(m, t, fft_u, fft_v ):
    champu, champv = ftau(m, t, fft_u, fft_v)

    champu_fft = fft(champu, axis=0)
    champv_fft = fft(champv, axis=0)
    champu_fft[0, :].fill(0)  # * champu_fft[0, :]
    champv_fft[0, :].fill(0)  # * champv_fft[0, :]
    champu = ifft(champu_fft / (1j * m.ktau), axis=0)
    champv = ifft(champv_fft / (1j * m.ktau), axis=0)

    C1u = fft_u + m.epsilon * champu
    C1v = fft_v + m.epsilon * champv

    return C1u, C1v

class DataSet:
    """ Class with initial data """

    def __init__(self, xmin, xmax, N, epsilon, Tfinal):

        self.N = N
        self.epsilon = epsilon

        self.k = 2 * pi / (xmax - xmin) * array([i if i < N/2 else -N+i for i in range(N)])
        self.T = 2 * pi
        self.Tfinal = Tfinal

        dx = (xmax - xmin) / N
        x = linspace(xmin, xmax - dx, N)[None]
        self.x = ravel(x)
        self.dx = x[0, 1] - x[0, 0]

        phi = (1 + 1j) * cos(x)
        gamma = (1 - 1j) * sin(x)

        self.sigma = 1
        self.llambda = -1

        self.u = phi - 1j * ifft((1 + epsilon * self.k ** 2) ** (-1 / 2) * fft(gamma))
        self.v = conj(phi) - 1j * ifft((1 + epsilon * self.k ** 2) ** (-1 / 2) * fft(conj(gamma)))

class MicMac:
    def __init__(self, data, ntau ):

        assert isinstance(data, DataSet)
        self.data = data
        self.ntau = ntau
        T = self.data.T
        ktau = 2 * pi / T * concatenate((arange(ntau // 2),
                arange(ntau // 2 - ntau, 0)), axis=0)
        tau = T * arange(ntau) / ntau
        tau = tau[:, newaxis]

        self.ktau = ktau[:, newaxis]
        self.ktau[0, 0] = 1
        self.matr = exp(1j * tau)
        self.conjmatr = exp(-1j * tau)
        k = self.data.k
        epsilon = self.data.epsilon
        if epsilon > 0:
            self.A1 = (sqrt(1 + epsilon * k ** 2) - 1) / epsilon
            self.A2 = (1 + epsilon * k ** 2) ** (-1 / 2)
        else:
            self.A1 = 0.5 * k ** 2
            self.A2 = 0 * k + 1

        self.llambda = self.data.llambda
        self.epsilon = self.data.epsilon
        self.sigma   = 1

    def run(self, dt):

        Tfinal = self.data.Tfinal
        N = self.data.N
        T = self.data.T
        k = self.data.k

        u = self.data.u
        v = self.data.v

        epsilon = self.data.epsilon

        dx = self.data.dx
        ktau = self.ktau

        k = self.data.k
        A1 = self.A1
        A2 = self.A2

        matr     = self.matr
        conjmatr = self.conjmatr
        ntau     = self.ntau

        t = 0
        iter = 0
        fft_u0 = fft(u)
        fft_v0 = fft(v)
        fft_ubar, fft_vbar, fft_ug, fft_vg = init_2(self, 0, fft_u0, fft_v0)

        while t < Tfinal:
            iter = iter + 1
            dt = min(Tfinal-t, dt)

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_2(self, t, fft_ubar, fft_vbar, fft_ug, fft_vg)
            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, self.ntau) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, self.ntau) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_2(self, t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12)

            fft_ubar = fft_ubar + dt * champubaru
            fft_vbar = fft_vbar + dt * champubarv
            fft_ug = fft_ug + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, self.ntau) \
                     - epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) + dt * champmoyu
            fft_vg = fft_vg + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, self.ntau) \
                     - epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) + dt * champmoyv

            t = t + dt
            C1u, C1v = C1(self, t, fft_ubar, fft_vbar)
            uC1eval = reconstr(C1u, t / epsilon, T, self.ntau)
            vC1eval = reconstr(C1v, t / epsilon, T, self.ntau)
            fft_u = uC1eval + fft_ug
            fft_v = vC1eval + fft_vg

        fft_u = exp(1j * sqrt(1 + epsilon * k ** 2) * t / epsilon) * fft_u
        fft_v = exp(1j * sqrt(1 + epsilon * k ** 2) * t / epsilon) * fft_v

        u = ifft(fft_u)
        v = ifft(fft_v)

        return u, v


def reconstr_x(u, x, xmin, xmax):
    N = shape(u)[1]
    L = xmax - xmin
    UA = 0.0
    v = fft(u) / N
    for jj in range(N // 2):
        vv = v[:, jj].reshape(shape(v)[0], 1)
        UA = UA + vv * exp(1j * 2 * pi / L * jj * (x - xmin))
    for jj in range(N // 2, N):
        vv = v[:, jj].reshape(shape(v)[0], 1)
        UA = UA + vv * exp(1j * 2 * pi / L * (jj - N) * (x - xmin))

    return UA

def erreur(u, v, epsilon):
    # ici la reference a été calculee par micmac

    str0 = 'test/donnees_data3_128_micmac/'
    str3 = 'donnee_'
    str5 = '.txt'
    str4 = {
    10:       '10',
    5:        '5',
    2.5:      '2_5',
    1:        '1',
    0.5:      '0_5',
    0.2:      '0_2',
    0.25:     '0_25',
    0.1:      '0_1',
    0.05:     '0_05',
    0.025:    '0_025',
    0.01:     '0_01',
    0.005:    '0_005',
    0.0025:   '0_0025',
    0.001:    '0_001',
    0.0005:   '0_0005',
    0.00025:  '0_00025',
    0.0001:   '0_0001',
    0.00005:  '0_00005',
    0.000025: '0_000025',
    0.00001:  '0_00001',
    0.000005: '0_000005',
    0.0000025: '0_0000025',
    0.000001: '0_000001'

    }
    fichier = os.path.join(str0, str3 + str4[epsilon] + str5)

    uv = zeros((4, 128))
    with open(fichier, "r") as azerty:

        for j in range(128):
            uv[0, j] = float(azerty.read(25))
            uv[1, j] = float(azerty.read(25))
            uv[2, j] = float(azerty.read(25))
            uv[3, j] = float(azerty.read(25))
            azerty.read(1)

    NN = shape(u)[1]
    xmin = 0
    xmax = 2 * pi
    T = 2 * pi
    t = 0.25

    dx = (xmax - xmin) / NN
    x = linspace(xmin, xmax - dx, NN)[None]
    dx = x[0, 1] - x[0, 0]
    k = 2 * pi / (xmax - xmin) * concatenate((arange(0, NN / 2, 1),
                                                 arange(NN / 2 - NN, 0, 1)), axis=0)

    if shape(uv)[1] != NN:
        uv = reconstr_x(uv, x, xmin, xmax)

    uvu = uv[0, :] + 1j * uv[1, :]
    uvv = uv[2, :] + 1j * uv[3, :]

    refH1 = sqrt(dx * norm(ifft(1j * sqrt(1 + k ** 2) * fft(uvu))) ** 2 
               + dx * norm(ifft(1j * sqrt(1 + k ** 2) * fft(uvv))) ** 2)
    
    err = (sqrt(dx * norm(ifft(1j * sqrt(1 + k ** 2) * fft(u - uvu))) ** 2 
              + dx * norm(ifft(1j * sqrt(1 + k ** 2) * fft(v - uvv))) ** 2)) / refH1

    return err

xmin    = 0
xmax    = 2 * pi
T       = 2 * pi
N       = 256
ntau    = 128
Tfinal  = 0.25
epsilon = 0.1

data = DataSet(xmin, xmax, N, epsilon, Tfinal)

dt = 2 ** (-3) * Tfinal / 16

m = MicMac(data, ntau)
start = time.time()
u, v = m.run(dt)
stop = time.time()
print(f'time = {stop-start}')
print(erreur(u, v, epsilon))
