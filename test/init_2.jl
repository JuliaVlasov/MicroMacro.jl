function init_2(t, fft_u0, fft_v0, A1, A2, matr, conjmatr, sigma,
           llambda, Ktaubis, epsilon, Ntaumm)

    champu, champv = ftau(t, fft_u0, fft_v0, A1, A2, matr, conjmatr, sigma, llambda)
    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[0, :] = 0 * champu_fft[0, :]
    champv_fft[0, :] = 0 * champv_fft[0, :]
    champu = ifft(champu_fft / (1im * Ktaubis), dims=1)
    champv = ifft(champv_fft / (1im * Ktaubis), dims=1)
    fft_ubar = fft_u0 - epsilon * champu[0, :]
    fft_vbar = fft_v0 - epsilon * champv[0, :]

    C1u, C1v = C1(t, fft_ubar, fft_vbar, A1, A2, matr,
                  conjmatr, sigma, llambda, Ktaubis, epsilon, Ntaumm)

    fft_ug = fft_u0 - C1u[0, :]
    fft_vg = fft_v0 - C1v[0, :]
    return fft_ubar, fft_vbar, fft_ug, fft_vg

end
