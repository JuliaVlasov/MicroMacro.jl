function champs_4(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr,
                  conjmatr, sigma, llambda, Ktaubis, epsilon, Ntaumm)


    h1u, h1v = ftau(t, 
		    fft_ubar, fft_vbar, 
		    A1, A2, 
		    matr, conjmatr, 
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

    dttu0u = champu_fft[1, :] / Ntaumm
    dttu0v = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth1u = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth1v = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu, champv = ftau(t, 
                          fft_ubar .+ h1u, fft_vbar .+ h1v, 
			  A1, A2, matr,
                          conjmatr, sigma, llambda)
    h2u = champu .- dth1u
    h2v = champv .- dth1v

    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    dtu1u = champu_fft[1, :] / Ntaumm
    dtu1v = champv_fft[1, :] / Ntaumm

    h2u_fft[1, :] .= 0.0
    h2v_fft[1, :] .= 0.0

    h2u = epsilon * ifft(h2u_fft ./ (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = d2uftau(t, fft_ubar, fft_vbar, 
                               dtu0u, dtu0v, dtu0u, dtu0v, 
                               A1, A2, matr, conjmatr, sigma,
                               llambda, Ntaumm)

    champu2, champv2 = dtduftau(t, 
				fft_ubar, fft_vbar, 
				dtu0u, dtu0v, 
                                A1, A2, 
				matr, conjmatr, 
				sigma, llambda, Ntaumm)

    champu3, champv3 = duftau(t, 
			      fft_ubar, fft_vbar, 
			      dttu0u, dttu0v, 
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu4, champv4 = d2tftau(t, 
			       fft_ubar, fft_vbar, 
			       A1, A2, 
                               matr, conjmatr, 
			       sigma, llambda, Ntaumm)

    champu = champu1 .+ 2 * champu2 .+ champu3 .+ champu4
    champv = champv1 .+ 2 * champv2 .+ champv3 .+ champv4

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    dtth1u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dtth1v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v,
                              dtu1u .+ dth1u, 
			      dtu1v .+ dth1v, 
			      A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v, 
			      A1, A2,
                              matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .- dtth1u
    champv = champv1 .+ champv2 .- dtth1v

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth2u = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth2v = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    h3u, h3v = ftau(t, 
		    fft_ubar .+ h2u, 
		    fft_vbar .+ h2v, 
		    A1, A2, 
		    matr, conjmatr, 
		    sigma, llambda)

    h3u = h3u .- dth2u
    h3v = h3v .- dth2v

    h3u_fft = fft(h3u, dims=1)
    h3v_fft = fft(h3v, dims=1)

    h3u_fft[1, :] .= 0 
    h3v_fft[1, :] .= 0 

    dtauh3u = ifft(h3u_fft, dims=1)
    dtauh3v = ifft(h3v_fft, dims=1)

    h3u = epsilon * ifft(h3u_fft ./ (1im * Ktaubis), dims=1)
    h3v = epsilon * ifft(h3v_fft ./ (1im * Ktaubis), dims=1)

    C3u = fft_ubar .+ h3u
    C3v = fft_vbar .+ h3v

    ffu, ffv = ftau(t, 
		    C3u .+ fft_ug, 
		    C3v .+ fft_vg, 
		    A1, A2, 
		    matr, conjmatr,
                    sigma, llambda)

    champu, champv = ftau(t, C3u, C3v, A1, A2, matr, conjmatr, 
			  sigma, llambda)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champubaru = champu_fft[1, :] / Ntaumm
    champubarv = champv_fft[1, :] / Ntaumm

    champu1, champv1 = duftau(t, fft_ubar, fft_vbar,
                              champubaru, champubarv, A1, A2, 
                              matr, conjmatr, sigma, llambda,
                              Ntaumm)

    champu2, champv2 = dtftau(t, fft_ubar, fft_vbar, A1, A2,
                              matr, conjmatr, sigma, llambda, Ntaumm)

    champu_fft = fft(champu1 .+ champu2, dims=1)
    champv_fft = fft(champv1 .+ champv2, dims=1)

    dttu0baru = champu_fft[1, :] / Ntaumm
    dttu0barv = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth1baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth1barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v,
                              champubaru .+ dth1baru, 
			      champubarv .+ dth1barv,
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v, 
			      A1, A2,
                              matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu_fft = fft(champu1 .+ champu2, dims=1)
    champv_fft = fft(champv1 .+ champv2, dims=1)

    dttu1baru = champu_fft[1, :] / Ntaumm
    dttu1barv = champv_fft[1, :] / Ntaumm

    champu1, champv1 = d2uftau(t, 
			       fft_ubar, fft_vbar, 
			       dtu0u, dtu0v, 
                               champubaru, champubarv, 
			       A1, A2, matr, conjmatr,
                               sigma, llambda, Ntaumm)

    champu2, champv2 = duftau(t, 
			      fft_ubar, fft_vbar, 
			      dttu0baru, dttu0barv, 
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda,
                              Ntaumm)

    champu3, champv3 = dtduftau(t, 
				fft_ubar, fft_vbar, 
				dtu0u, dtu0v, 
                                A1, A2, 
				matr, conjmatr, 
				sigma, llambda, Ntaumm)

    champu4, champv4 = dtduftau(t, 
				fft_ubar, fft_vbar, 
				champubaru, champubarv,
                                A1, A2, 
				matr, conjmatr, 
				sigma, llambda,
                                Ntaumm)

    champu5, champv5 = d2tftau(t, 
			       fft_ubar, fft_vbar, 
			       A1, A2, 
			       matr, conjmatr, 
                               sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .+ champu3 .+ champu4 .+ champu5
    champv = champv1 .+ champv2 .+ champv3 .+ champv4 .+ champv5

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    dtttu0baru = champu_fft[1, :] / Ntaumm
    dtttu0barv = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0

    dtth1baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dtth1barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = d3uftau(t, 
			       fft_ubar, fft_vbar, 
			       dtu0u, dtu0v, 
                               dtu0u, dtu0v, 
			       champubaru, champubarv, 
			       A1, A2, 
			       matr, conjmatr, 
			       sigma, llambda, Ntaumm)

    champu2, champv2 = dtd2uftau(t, 
				 fft_ubar, fft_vbar, 
				 dtu0u, dtu0v, 
                                 dtu0u, dtu0v, 
				 A1, A2, 
				 matr, conjmatr, 
				 sigma, llambda, Ntaumm)

    champu3, champv3 = d2uftau(t, 
			       fft_ubar, fft_vbar, 
                               dttu0baru, dttu0barv, 
			       dtu0u, dtu0v, 
			       A1, A2,
                               matr, conjmatr, 
			       sigma, llambda, Ntaumm)

    champu4, champv4 = dtd2uftau(t, 
				 fft_ubar, fft_vbar, 
				 dtu0u, dtu0v,
                                 champubaru, champubarv, 
				 A1, A2, matr, conjmatr,
                                 sigma, llambda, Ntaumm)

    champu5, champv5 = d2tduftau(t, 
				 fft_ubar, fft_vbar, 
				 dtu0u, dtu0v,
                                 A1, A2, 
				 matr, conjmatr, 
				 sigma, llambda, Ntaumm)

    champu6, champv6 = dtduftau(t, fft_ubar, fft_vbar,
                                dttu0baru, dttu0barv, 
                                A1, A2, 
				matr, conjmatr, 
				sigma, llambda,
                                Ntaumm)

    champu7, champv7 = d2uftau(t, 
			       fft_ubar, fft_vbar,
                               dttu0u, dttu0v, 
			       champubaru, champubarv, 
                               A1, A2, 
			       matr, conjmatr,
                               sigma, llambda, Ntaumm)

    champu8, champv8 = dtduftau(t, 
				fft_ubar, fft_vbar,
                                dttu0u, dttu0v, 
				A1, A2, 
				matr, conjmatr, 
				sigma, llambda, Ntaumm)

    champu9, champv9 = duftau(t, 
			      fft_ubar, fft_vbar, 
                              dtttu0baru, dtttu0barv, 
			      A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda,
                              Ntaumm)

    champu10, champv10 = d3tftau(t, 
				 fft_ubar, fft_vbar, 
				 A1, A2,
                                 matr, conjmatr, 
				 sigma, llambda, Ntaumm)

    champu11, champv11 = d2tduftau(t, 
				   fft_ubar, fft_vbar, 
                                   champubaru, champubarv, 
				   A1, A2, 
				   matr, conjmatr, 
				   sigma, llambda, Ntaumm)

    champu = (champu1 .+ champu2 .+ 2 * champu3 .+ 2 * champu4 
                     .+ 2 * champu5 .+ 2 * champu6 .+ champu7 
		     .+ champu8 .+ champu9 .+ champu10 .+ champu11)

    champv = (champv1 .+ champv2 .+ 2 * champv3 .+ 2 * champv4 
             .+ 2 * champv5 .+ 2 * champv6 .+ champv7 .+ champv8 
	     .+ champv9 .+ champv10 .+ champv11)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dttth1baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dttth1barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = d2uftau(t, 
			       fft_ubar .+ h1u, fft_vbar .+ h1v,
                               dtu1u .+ dth1u, dtu1v .+ dth1v,
                               champubaru .+ dth1baru, 
			       champubarv .+ dth1barv,
                               A1, A2, 
			       matr, conjmatr, 
			       sigma, llambda, Ntaumm)

    champu2, champv2 = duftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v,
                              dttu1baru .+ dtth1baru, 
			      dttu1barv .+ dtth1barv,
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu3, champv3 = dtduftau(t, 
				fft_ubar .+ h1u, 
				fft_vbar .+ h1v,
                                dtu1u .+ dth1u, 
				dtu1v .+ dth1v, 
				A1, A2, 
				matr, conjmatr, 
				sigma, llambda, Ntaumm)

    champu4, champv4 = dtduftau(t, 
				fft_ubar .+ h1u, fft_vbar .+ h1v,
                                champubaru .+ dth1baru,
                                champubarv .+ dth1barv, 
				A1, A2, 
				matr, conjmatr, 
				sigma, llambda,
                                Ntaumm)

    champu5, champv5 = d2tftau(t, 
			       fft_ubar .+ h1u, fft_vbar .+ h1v, 
			       A1, A2,
                               matr, conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .+ champu3 .+ champu4 .+ champu5 .- dttth1baru
    champv = champv1 .+ champv2 .+ champv3 .+ champv4 .+ champv5 .- dttth1barv

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0

    dtth2baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dtth2barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v,
                              champubaru .+ dth1baru, 
			      champubarv .+ dth1barv,
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar .+ h1u, 
			      fft_vbar .+ h1v, 
			      A1, A2,
                              matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu_fft = fft(champu1 .+ champu2 .- dtth1baru, dims=1)
    champv_fft = fft(champv1 .+ champv2 .- dtth1barv, dims=1)

    dttu0baru = champu_fft[1, :] / Ntaumm
    dttu0barv = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth2baru = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth2barv = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, 
			      fft_ubar .+ h2u, 
			      fft_vbar .+ h2v,
                              champubaru .+ dth2baru, 
			      champubarv .+ dth2barv,
                              A1, A2, 
			      matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, 
			      fft_ubar .+ h2u, 
			      fft_vbar .+ h2v, 
			      A1, A2,
                              matr, conjmatr, 
			      sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2 .- dtth2baru
    champv = champv1 .+ champv2 .- dtth2barv

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0

    dtC3u = champubaru .+ epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dtC3v = champubarv .+ epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    champgu = ffu .- dtauh3u .- dtC3u
    champgv = ffv .- dtauh3v .- dtC3v

    champgu_fft = fft(champgu, dims=1)
    champgv_fft = fft(champgv, dims=1)

    champmoyu = champgu_fft[1, :] / Ntaumm
    champmoyv = champgv_fft[1, :] / Ntaumm

    champgu_fft[1, :] .= 0 
    champgv_fft[1, :] .= 0

    ichampgu = ifft(champgu_fft ./ (1im * Ktaubis), dims=1)
    ichampgv = ifft(champgv_fft ./ (1im * Ktaubis), dims=1)

    champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv

end
