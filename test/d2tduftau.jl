function d2tduftau(t, fft_u, fft_v, fft_du, fft_dv, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm):
    # %% attention ici je n'ai code' que le cas sigma=1
    sigma = 1

    u = ifft(exp(1j * t * A1) * fft_u * matr)
    v = ifft(exp(1j * t * A1) * fft_v * matr)
    dtu = ifft(exp(1j * t * A1) * (1j * A1) * fft_u) * matr
    dtv = ifft(exp(1j * t * A1) * (1j * A1) * fft_v) * matr
    d2tu = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_u) * matr
    d2tv = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_v) * matr

    du = ifft(exp(1j * t * A1) * fft_du * matr)
    dv = ifft(exp(1j * t * A1) * fft_dv * matr)
    dtdu = ifft(exp(1j * t * A1) * (1j * A1) * fft_du) * matr
    dtdv = ifft(exp(1j * t * A1) * (1j * A1) * fft_dv) * matr
    d2tdu = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_du) * matr
    d2tdv = ifft(exp(1j * t * A1) * (-A1 ** 2) * fft_dv) * matr

    z = (u + conj(v)) / 2
    dtz = (dtu + conj(dtv)) / 2
    d2tz = (d2tu + conj(d2tv)) / 2
    dz = (du + conj(dv)) / 2
    dtdz = (dtdu + conj(dtdv)) / 2
    d2tdz = (d2tdu + conj(d2tdv)) / 2

    fz1 = 2 * dz * conj(z) * d2tz + 2 * z * conj(dz) * d2tz + 2 * z * conj(
        z) * d2tdz + 4 * dtz * dtdz * conj(z) + 2 * dtz ** 2 * conj(dz) + 4 * dtdz * conj(
        dtz) * z + 4 * dtz * conj(dtdz) * z + 4 * dtz * conj(dtz) * dz + 2 * z * dz * conj(
        d2tz) + z ** 2 * conj(d2tdz)

    champu1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    fz1 = 2 * dtz * conj(z) * dz + 2 * z * conj(dtz) * dz + 2 * abs(
        z) ** 2 * dtdz + 2 * z * dtz * conj(dz) + z ** 2 * conj(dtdz)
    champu2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * fz1)
    champv2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * conj(fz1))

    fz1 = 2 * abs(z) ** 2 * dz + z ** 2 * conj(dz)
    champu3 = -1j * llambda * A2 * exp(-1j * t * A1) * (-A1 ** 2) * fft(conjmatr * fz1)
    champv3 = -1j * llambda * A2 * exp(-1j * t * A1) * (-A1 ** 2) * fft(conjmatr * conj(fz1))

    champu = champu1 + 2 * champu2 + champu3
    champv = champv1 + 2 * champv2 + champv3

    return champu, champv

end
