function ichampf(t, fft_u, fft_v, A1, A2, Ntaumm, matr, conjmatr, Ktaubis, llambda, sigma)

    u = ifft(exp(1j * t * A1) * fft_u) * matr
    v = ifft(exp(1j * t * A1) * fft_v) * matr

    z = (u + conj(v)) / 2

    fz1 = abs(z) ** (2 * sigma) * z
    champu = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * fz1)
    champv = -1j * llambda * A2 * exp(-1j * t * A1) * fft(conjmatr * conj(fz1))

    champu_fft = fft(champu, axis=0)
    champv_fft = fft(champv, axis=0)

    champumoy = champu_fft[0, :] / Ntaumm
    champvmoy = champv_fft[0, :] / Ntaumm

    champu_fft[0, :] = 0 * champu_fft[0, :]
    champv_fft[0, :] = 0 * champv_fft[0, :]
    champu = ifft(champu_fft / (1j * Ktaubis), axis=0)
    champv = ifft(champv_fft / (1j * Ktaubis), axis=0)

    return champu, champv, champumoy, champvmoy

end
