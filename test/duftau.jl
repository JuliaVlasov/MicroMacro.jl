function duftau(t, fft_u, fft_v, fft_du, fft_dv, A1, A2, 
                matr, conjmatr, sigma, llambda, Ntaumm)

    sigma = 1

    u = ifft(exp(1j * t * A1) * fft_u * matr)
    v = ifft(exp(1j * t * A1) * fft_v * matr)

    du = ifft(exp(1j * t * A1) * fft_du * matr)
    dv = ifft(exp(1j * t * A1) * fft_dv * matr)

    z = (u + conj(v)) / 2
    dz = (du + conj(dv)) / 2

    fz1 = 2 * abs(z) ** 2. * dz + z ** 2. * conj(dz)

    champu = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    return champu, champv

end
