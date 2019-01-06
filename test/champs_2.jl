function champs_2(t, fft_ubar, fft_vbar, fft_ug, fft_vg,
                  A1, A2, matr, conjmatr, sigma, llambda, 
                  ktau, epsilon, Ntaumm)

    champu, champv = ftau(t, fft_ubar, fft_vbar, 
                     A1, A2, matr, conjmatr, sigma, llambda)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    dtauh1u = ifft(champu_fft, 1)
    dtauh1v = ifft(champv_fft, 1)

    h1u = epsilon * ifft(champu_fft ./ (1im * ktau), 1)
    h1v = epsilon * ifft(champv_fft ./ (1im * ktau), 1)

    C1u = fft_ubar .+ h1u
    C1v = fft_vbar .+ h1v

    ffu, ffv = ftau(t,
                    C1u .+ fft_ug,
                    C1v .+ fft_vg,
                    A1, A2, matr, conjmatr,
                    sigma, llambda)

    champu, champv = ftau(t, C1u, C1v, A1, A2, matr, conjmatr, sigma, llambda)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champubaru = champu_fft[1, :] / Ntaumm
    champubarv = champv_fft[1, :] / Ntaumm

    champu1, champv1 = duftau(t, fft_ubar, fft_vbar,
                              champubaru, champubarv,
                              A1, A2, matr, conjmatr, sigma, llambda,
                              Ntaumm)

    champu2, champv2 = dtftau(t, fft_ubar, fft_vbar,
                              A1, A2, matr, conjmatr,
                              sigma, llambda, Ntaumm)

    champu_fft = fft(champu1 .+ champu2, 1)
    champv_fft = fft(champv1 .+ champv2, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    dtC1u = champubaru .+ epsilon * ifft(champu_fft / (1im * ktau), 1)
    dtC1v = champubarv .+ epsilon * ifft(champv_fft / (1im * ktau), 1)

    champgu = ffu .- dtauh1u .- dtC1u
    champgv = ffv .- dtauh1v .- dtC1v

    champgu_fft = fft(champgu, 1)
    champgv_fft = fft(champgv, 1)

    champmoyu = champgu_fft[1, :] / Ntaumm
    champmoyv = champgv_fft[1, :] / Ntaumm

    champgu_fft[1, :] .= 0.0 
    champgv_fft[1, :] .= 0.0

    ichampgu = ifft(champgu_fft ./ (1im * ktau), 1)
    ichampgv = ifft(champgv_fft ./ (1im * ktau), 1)

    champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv

end
