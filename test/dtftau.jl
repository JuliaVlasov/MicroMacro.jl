function dtftau(t, fft_u, fft_v, A1, A2, 
                matr, conjmatr, sigma, llambda, Ntaumm)

    # attention ici je n'ai code' que le cas sigma=1

    sigma = 1

    u = ifft(exp(1j * t * A1) * fft_u * matr)
    v = ifft(exp(1j * t * A1) * fft_v * matr)
    du = ifft(exp(1j * t * A1) * (1j * A1) * fft_u) * matr
    dv = ifft(exp(1j * t * A1) * (1j * A1) * fft_v) * matr

    z = (u + conj(v)) / 2
    dz = (du + conj(dv)) / 2

    fz1 = 2 * abs(z) ** 2 * dz + z ** 2 * conj(dz)
    champu1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv1 = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    fz1 = abs(z) ** 2 * z
    champu2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * fz1)
    champv2 = -1j * llambda * A2 * exp(-1j * t * A1) * (-1j * A1) * fft(conjmatr * conj(fz1))

    champu = champu1 + champu2
    champv = champv1 + champv2

    return champu, champv

end
