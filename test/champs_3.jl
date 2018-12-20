function champs_3(t, fft_ubar, fft_vbar, fft_ug, fft_vg, 
                  A1, A2, matr, conjmatr, sigma, llambda, 
                  Ktaubis, epsilon, Ntaumm)

    h1u, h1v = ftau(t, fft_ubar, fft_vbar, 
                    A1, A2, matr, conjmatr, 
                    sigma, llambda)

    h1u_fft = fft(h1u, dims=1)
    h1v_fft = fft(h1v, dims=1)

    dtu0u = h1u_fft[1, :] / Ntaumm
    dtu0v = h1v_fft[1, :] / Ntaumm

    h1u_fft[1, :] .= 0 
    h1v_fft[1, :] .= 0

    h1u = epsilon * ifft(h1u_fft ./ (1im * Ktaubis), dims=1)
    h1v = epsilon * ifft(h1v_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar, fft_vbar, 
			      dtu0u, dtu0v, 
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar, fft_vbar, 
			      A1, A2, 
			      matr, conjmatr, 
                              sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2
    champv = champv1 .+ champv2

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    dth1u = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth1v = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    h2u, h2v = ftau(t, 
		    fft_ubar .+ h1u, 
                    fft_vbar .+ h1v, 
                    A1, A2, matr,
                    conjmatr, sigma, llambda)

    h2u = h2u .- dth1u
    h2v = h2v .- dth1v

    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)

    h2u_fft[1, :] .= 0.0
    h2v_fft[1, :] .= 0.0

    dtauh2u = ifft(h2u_fft, dims=1)
    dtauh2v = ifft(h2v_fft, dims=1)

    h2u = epsilon * ifft(h2u_fft ./ (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft ./ (1im * Ktaubis), dims=1)

    C2u = fft_ubar .+ h2u
    C2v = fft_vbar .+ h2v

    ffu, ffv = ftau(t, 
		    C2u .+ fft_ug, 
		    C2v .+ fft_vg,
                    A1, A2, 
		    matr, conjmatr,
                    sigma, llambda)

    champu, champv = ftau(t, 
			  C2u, C2v, 
			  A1, A2, 
			  matr, conjmatr, 
			  sigma, llambda)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champubaru = champu_fft[1, :] / Ntaumm
    champubarv = champv_fft[1, :] / Ntaumm

    champu1, champv1 = duftau(t, fft_ubar, fft_vbar, 
                              champubaru, champubarv, 
                              A1, A2, matr, conjmatr, 
                              sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, fft_ubar, fft_vbar, 
                              A1, A2, matr, conjmatr, 
                              sigma, llambda, Ntaumm)

    champu_fft = fft(champu1 .+ champu2, dims=1)
    champv_fft = fft(champv1 .+ champv2, dims=1)

    dttu0baru = champu_fft[1, :] / Ntaumm
    dttu0barv = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0.0 
    champv_fft[1, :] .= 0.0

    dth1baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth1barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = d2uftau(t, fft_ubar, fft_vbar, dtu0u, dtu0v, 
                               champubaru, champubarv, A1, A2, matr, 
                               conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = duftau(t, fft_ubar, fft_vbar, dttu0baru, dttu0barv, 
                              A1, A2, matr, conjmatr, sigma, llambda,
                              Ntaumm)

    champu3, champv3 = dtduftau(t, fft_ubar, fft_vbar, dtu0u, dtu0v, 
                                A1, A2, matr, conjmatr, sigma, llambda, 
				Ntaumm)

    champu4, champv4 = dtduftau(t, fft_ubar, fft_vbar, 
				champubaru, champubarv, 
                                A1, A2, matr, conjmatr, sigma, llambda,
                                Ntaumm)

    champu5, champv5 = d2tftau(t, fft_ubar, fft_vbar, A1, A2, 
			       matr, conjmatr, 
                               sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .+ champu3 .+ champu4 .+ champu5
    champv = champv1 .+ champv2 .+ champv3 .+ champv4 .+ champv5

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dtth1baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dtth1barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v,
                              champubaru .+ dth1baru, 
			      champubarv .+ dth1barv,
                              A1, A2, matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v, 
			      A1, A2,
                              matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .- dtth1baru
    champv = champv1 .+ champv2 .- dtth1barv

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    dtC2u = champubaru .+ epsilon * ifft(champu_fft ./ (1im * Ktaubis), 
					 dims=1)
    dtC2v = champubarv .+ epsilon * ifft(champv_fft ./ (1im * Ktaubis), 
					 dims=1)

    champgu = ffu .- dtauh2u .- dtC2u
    champgv = ffv .- dtauh2v .- dtC2v

    champgu_fft = fft(champgu, dims=1)
    champgv_fft = fft(champgv, dims=1)

    champmoyu = champgu_fft[1, :] / Ntaumm
    champmoyv = champgv_fft[1, :] / Ntaumm

    champgu_fft[1, :] .= 0.0
    champgv_fft[1, :] .= 0.0

    ichampgu = ifft(champgu_fft ./ (1im * Ktaubis), dims=1)
    ichampgv = ifft(champgv_fft ./ (1im * Ktaubis), dims=1)

    champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv

end
