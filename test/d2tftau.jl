function d2tftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm):

    sigma = 1

    u = ifft(exp(1j * t * A1) * fft_u * matr)
    v = ifft(exp(1j * t * A1) * fft_v * matr)
    du = ifft(exp(1j * t * A1) * (1j * A1) * fft_u) * matr
    dv = ifft(exp(1j * t * A1) * (1j * A1) * fft_v) * matr
    d2u = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_u) * matr
    d2v = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_v) * matr

    z = (u + conj(v)) / 2
    dz = (du + conj(dv)) / 2
    d2z = (d2u + conj(d2v)) / 2

    fz1 = 2 * abs(z) ** 2 * d2z + 2 * dz ** 2 * conj(z) + 4 * abs(dz) ** 2 * z + z ** 2 * conj(d2z)
    champu1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    fz1 = 2 * abs(z) ** 2 * dz + z ** 2 * conj(dz)
    champu2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * fz1)
    champv2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * conj(fz1))

    fz1 = abs(z) ** 2. * z
    champu3 = -1j * llambda * A2 * exp(-1j * t * A1) * (-A1 ** 2) * fft(conjmatr * fz1)
    champv3 = -1j * llambda * A2 * exp(-1j * t * A1) * (-A1 ** 2) * fft(conjmatr * conj(fz1))

    champu = champu1 + 2 * champu2 + champu3
    champv = champv1 + 2 * champv2 + champv3

    return champu, champv

end
