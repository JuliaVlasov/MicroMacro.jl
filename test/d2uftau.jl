function d2uftau(t, fft_u, fft_v, fft_du, fft_dv, fft_du2, fft_dv2, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm):

    sigma = 1

    u = ifft(exp(1j * t * A1) * fft_u) * matr
    v = ifft(exp(1j * t * A1) * fft_v) * matr

    du = ifft(exp(1j * t * A1) * fft_du) * matr
    dv = ifft(exp(1j * t * A1) * fft_dv) * matr

    du2 = ifft(exp(1j * t * A1) * fft_du2) * matr
    dv2 = ifft(exp(1j * t * A1) * fft_dv2) * matr

    z = (u + conj(v)) / 2
    dz = (du + conj(dv)) / 2
    dz2 = (du2 + conj(dv2)) / 2

    fz1 = 2 * z * dz * conj(dz2) + 2 * z * conj(dz) * dz2 + 2 * conj(z) * dz * dz2
    champu = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    return champu, champv

end
