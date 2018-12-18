from duftau import duftau
from dtftau import dtftau
from ftau import ftau
import scipy as sp

def C2(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda, Ktaubis, epsilon, Ntaumm):
    h1u, h1v = ftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, axis=0)
    h1v_fft = fft(h1v, axis=0)

    dtu0u = h1u_fft[0, :] / Ntaumm
    dtu0v = h1v_fft[0, :] / Ntaumm

    h1u_fft[0, :] = 0 * h1u_fft[0, :]
    h1v_fft[0, :] = 0 * h1v_fft[0, :]

    h1u = epsilon * ifft(h1u_fft / (1j * Ktaubis), axis=0)
    h1v = epsilon * ifft(h1v_fft / (1j * Ktaubis), axis=0)

    champu1, champv1 = duftau(t, fft_u, fft_v, dtu0u, dtu0v, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    champu2, champv2 = dtftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    champu = champu1 + champu2
    champv = champv1 + champv2
    champu_fft = fft(champu, axis=0)
    champv_fft = fft(champv, axis=0)
    champu_fft[0, :] = 0 * champu_fft[0, :]
    champv_fft[0, :] = 0 * champv_fft[0, :]
    dth1u = epsilon * ifft(champu_fft / (1j * Ktaubis), axis=0)
    dth1v = epsilon * ifft(champv_fft / (1j * Ktaubis), axis=0)

    h2u, h2v = ftau(t, sp.tile(fft_u, (Ntaumm, 1)) + h1u, sp.tile(fft_v, (Ntaumm, 1)) + h1v, A1, A2, matr, conjmatr,
                    sigma, llambda)
    h2u = h2u - dth1u
    h2v = h2v - dth1v
    h2u_fft = fft(h2u, axis=0)
    h2v_fft = fft(h2v, axis=0)
    h2u_fft[0, :] = 0 * h2u_fft[0, :]
    h2v_fft[0, :] = 0 * h2v_fft[0, :]
    h2u = epsilon * ifft(h2u_fft / (1j * Ktaubis), axis=0)
    h2v = epsilon * ifft(h2v_fft / (1j * Ktaubis), axis=0)

    C2u = sp.tile(fft_u, (Ntaumm, 1)) + h2u
    C2v = sp.tile(fft_v, (Ntaumm, 1)) + h2v

    return C2u, C2v
