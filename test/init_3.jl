function init_3(t, fft_u0, fft_v0, A1, A2, matr, conjmatr,
           sigma, llambda, Ktaubis, epsilon, Ntaumm)

    h1u, h1v = ftau(t, fft_u0, fft_v0, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, axis=0)
    h1v_fft = fft(h1v, axis=0)

    dtu0u = h1u_fft[0, :] / Ntaumm
    dtu0v = h1v_fft[0, :] / Ntaumm

    h1u_fft[0, :] = 0 * h1u_fft[0, :]
    h1v_fft[0, :] = 0 * h1v_fft[0, :]

    h1u = epsilon * ifft(h1u_fft / (1j * Ktaubis), axis=0)
    h1v = epsilon * ifft(h1v_fft / (1j * Ktaubis), axis=0)

    w1u = fft_u0 - h1u[0, :]
    w1v = fft_v0 - h1v[0, :]

    h1u, h1v = ftau(t, w1u, w1v, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, axis=0)
    h1v_fft = fft(h1v, axis=0)

    dtu0u = h1u_fft[0, :] / Ntaumm
    dtu0v = h1v_fft[0, :] / Ntaumm

    h1u_fft[0, :] = 0 * h1u_fft[0, :]
    h1v_fft[0, :] = 0 * h1v_fft[0, :]

    h1u = epsilon * ifft(h1u_fft / (1j * Ktaubis), axis=0)
    h1v = epsilon * ifft(h1v_fft / (1j * Ktaubis), axis=0)

    champu1, champv1 = duftau(t, w1u, w1v, dtu0u, dtu0v,
                              A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, w1u, w1v, A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + champu2
    champv = champv1 + champv2
    champu_fft = fft(champu, axis=0)
    champv_fft = fft(champv, axis=0)
    champu_fft[0, :] = 0 * champu_fft[0, :]
    champv_fft[0, :] = 0 * champv_fft[0, :]
    dth1u = epsilon * ifft(champu_fft / (1j * Ktaubis), axis=0)
    dth1v = epsilon * ifft(champv_fft / (1j * Ktaubis), axis=0)

    h2u, h2v = ftau(t, w1u + h1u, w1v + h1v, A1, A2, matr, conjmatr, sigma,
                    llambda)
    h2u = h2u - dth1u
    h2v = h2v - dth1v
    h2u_fft = fft(h2u, axis=0)
    h2v_fft = fft(h2v, axis=0)
    h2u_fft[0, :] = 0 * h2u_fft[0, :]
    h2v_fft[0, :] = 0 * h2v_fft[0, :]
    h2u = epsilon * ifft(h2u_fft / (1j * Ktaubis), axis=0)
    h2v = epsilon * ifft(h2v_fft / (1j * Ktaubis), axis=0)

    fft_ubar = fft_u0 - h2u[0, :]
    fft_vbar = fft_v0 - h2v[0, :]

    C2u, C2v = C2(t, fft_ubar, fft_vbar, A1, A2, matr, conjmatr,
                  sigma, llambda, Ktaubis, epsilon, Ntaumm)

    fft_ug = fft_u0 - C2u[0, :]
    fft_vg = fft_v0 - C2v[0, :]

    return fft_ubar, fft_vbar, fft_ug, fft_vg

end
