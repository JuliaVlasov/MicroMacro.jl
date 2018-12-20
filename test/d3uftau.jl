function d3uftau(t, fft_u, fft_v, fft_du, fft_dv, fft_du2, fft_dv2, fft_du3,
            fft_dv3, A1, A2, matr, conjmatr, sigma, llambda,
            Ntaumm)

    sigma = 1

    u = ifft(exp(1im * t * A1) * fft_u) * matr
    v = ifft(exp(1im * t * A1) * fft_v) * matr

    du = ifft(exp(1im * t * A1) * fft_du) * matr
    dv = ifft(exp(1im * t * A1) * fft_dv) * matr

    du2 = ifft(exp(1im * t * A1) * fft_du2) * matr
    dv2 = ifft(exp(1im * t * A1) * fft_dv2) * matr

    du3 = ifft(exp(1im * t * A1) * fft_du3) * matr
    dv3 = ifft(exp(1im * t * A1) * fft_dv3) * matr

    z = (u + conj(v)) / 2
    dz = (du + conj(dv)) / 2
    dz2 = (du2 + conj(dv2)) / 2
    dz3 = (du3 + conj(dv3)) / 2

    fz1 = 2 * dz3 * dz * conj(dz2) + 2 * dz3 * conj(dz) * dz2 + 2 * conj(dz3) * dz * dz2
    champu = -1im * llambda * A2 * exp(-1im * t * A1) * fft(conjmatr * fz1)
    champv = -1im * llambda * A2 * exp(-1im * t * A1) * fft(conjmatr * conj(fz1))

    return champu, champv

end
