function init_2(t,
                fft_u0, fft_v0,
                A1, A2,
                matr, conjmatr,
                sigma, llambda,
                ktau, epsilon, ntau)


    champu, champv = ftau(t, fft_u0, fft_v0, A1, A2,
                          matr, conjmatr, sigma, llambda)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    champu = ifft(champu_fft ./ (1im * ktau), 1)
    champv = ifft(champv_fft ./ (1im * ktau), 1)

    fft_ubar = fft_u0 .- epsilon * transpose(champu[1, :])
    fft_vbar = fft_v0 .- epsilon * transpose(champv[1, :])

    println("fft_ubar:",size(fft_ubar))

    C1u, C1v = C1(t, fft_ubar, fft_vbar, A1, A2, matr,
                  conjmatr, sigma, llambda, ktau, epsilon, ntau)

    fft_ug = fft_u0 .- transpose(C1u[1, :])
    fft_vg = fft_v0 .- transpose(C1v[1, :])

    fft_ubar, fft_vbar, fft_ug, fft_vg

end
