function init_4(t,  
		fft_u0, fft_v0, 
		A1, A2, 
		matr, conjmatr,
                sigma, llambda, 
		Ktaubis, epsilon, Ntaumm)

    h1u, h1v = ftau(t, fft_u0, fft_v0, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, dims=1)
    h1v_fft = fft(h1v, dims=1)

    dtu0u = h1u_fft[1, :] / Ntaumm
    dtu0v = h1v_fft[1, :] / Ntaumm

    h1u_fft[1, :] .= 0 
    h1v_fft[1, :] .= 0 

    h1u = epsilon * ifft(h1u_fft / (1im * Ktaubis), dims=1)
    h1v = epsilon * ifft(h1v_fft / (1im * Ktaubis), dims=1)

    w1u = fft_u0 .- h1u[1, :]
    w1v = fft_v0 .- h1v[1, :]

    h1u, h1v = ftau(t, w1u, w1v, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, dims=1)
    h1v_fft = fft(h1v, dims=1)

    dtu0u = h1u_fft[1, :] / Ntaumm
    dtu0v = h1v_fft[1, :] / Ntaumm

    h1u_fft[1, :] .= 0 
    h1v_fft[1, :] .= 0 

    h1u = epsilon * ifft(h1u_fft / (1im * Ktaubis), dims=1)
    h1v = epsilon * ifft(h1v_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, w1u, w1v, dtu0u, dtu0v,
                              A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, w1u, w1v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 .+ champu2
    champv = champv1 .+ champv2

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth1u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dth1v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    h2u, h2v = ftau(t, w1u + h1u, w1v + h1v, A1, A2, matr, conjmatr, sigma,
                    llambda)

    h2u = h2u - dth1u
    h2v = h2v - dth1v

    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)

    h2u_fft[1, :] .= 0 
    h2v_fft[1, :] .= 0 

    h2u = epsilon * ifft(h2u_fft / (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft / (1im * Ktaubis), dims=1)

    w2u = fft_u0 .- h2u[1, :]
    w2v = fft_v0 .- h2v[1, :]

    h1u, h1v = ftau(t, w2u, w2v, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, dims=1)
    h1v_fft = fft(h1v, dims=1)

    dtu0u = h1u_fft[1, :] / Ntaumm
    dtu0v = h1v_fft[1, :] / Ntaumm

    h1u_fft[1, :] .= 0 
    h1v_fft[1, :] .= 0 

    h1u = epsilon * ifft(h1u_fft / (1im * Ktaubis), dims=1)
    h1v = epsilon * ifft(h1v_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, w2u, w2v, dtu0u, dtu0v,
                              A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, w2u, w2v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + champu2
    champv = champv1 + champv2

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    dttu0u = champu_fft[1, :] / Ntaumm
    dttu0v = champv_fft[1, :] / Ntaumm

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth1u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dth1v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    champu, champv = ftau(t, w2u + h1u, w2v + h1v, A1, A2, matr, conjmatr,
                          sigma, llambda)

    h2u = champu - dth1u
    h2v = champv - dth1v

    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    dtu1u = champu_fft[1, :] / Ntaumm
    dtu1v = champv_fft[1, :] / Ntaumm

    h2u_fft[1, :] .= 0 
    h2v_fft[1, :] .= 0 

    h2u = epsilon * ifft(h2u_fft / (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = d2uftau(t, w2u, w2v, dtu0u, dtu0v, dtu0u, dtu0v,
                               A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    champu2, champv2 = dtduftau(t, w2u, w2v, dtu0u, dtu0v,
                                A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    champu3, champv3 = duftau(t, w2u, w2v, dttu0u, dttu0v,
                              A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    champu4, champv4 = d2tftau(t, w2u, w2v,
                               A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + 2 * champu2 + champu3 + champu4
    champv = champv1 + 2 * champv2 + champv3 + champv4

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dtth1u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dtth1v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, w2u + h1u, w2v + h1v,
                              dtu1u + dth1u, dtu1v + dth1v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, w2u + h1u, w2v + h1v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + champu2 - dtth1u
    champv = champv1 + champv2 - dtth1v

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth2u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dth2v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    h3u, h3v = ftau(t, w2u + h2u, w2v + h2v, A1, A2, matr, conjmatr, sigma,
                    llambda)

    h3u = h3u - dth2u
    h3v = h3v - dth2v

    h3u_fft = fft(h3u, dims=1)
    h3v_fft = fft(h3v, dims=1)

    h3u_fft[1, :] .= 0 
    h3v_fft[1, :] .= 0 

    h3u = epsilon * ifft(h3u_fft / (1im * Ktaubis), dims=1)
    h3v = epsilon * ifft(h3v_fft / (1im * Ktaubis), dims=1)

    h2u, h2v = ftau(t, w2u + h1u, w2v + h1v, 
		    A1, A2, 
		    matr, conjmatr, 
		    sigma, llambda)

    h2u = h2u - dth1u
    h2v = h2v - dth1v

    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)

    h2u_fft[1, :] .= 0 
    h2v_fft[1, :] .= 0 

    h2u = epsilon * ifft(h2u_fft / (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = d2uftau(t, w2u, w2v, dtu0u, dtu0v, dtu0u, dtu0v,
                               A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtduftau(t, w2u, w2v, dtu0u, dtu0v,
                                A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    
    champu3, champv3 = duftau(t, w2u, w2v, dttu0u, dttu0v,
                              A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)
    
    champu4, champv4 = d2tftau(t, w2u, w2v,
                               A1, A2, matr, conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + 2 * champu2 + champu3 + champu4
    champv = champv1 + 2 * champv2 + champv3 + champv4

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dtth1u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dtth1v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, w2u + h1u, w2v + h1v,
                              dtu0u + dth1u, dtu0v + dth1v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu2, champv2 = dtftau(t, w2u + h1u, w2v + h1v, A1, A2, matr,
                              conjmatr, sigma, llambda, Ntaumm)

    champu = champu1 + champu2 - dtth1u
    champv = champv1 + champv2 - dtth1v

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    dth2u = epsilon * ifft(champu_fft / (1im * Ktaubis), dims=1)
    dth2v = epsilon * ifft(champv_fft / (1im * Ktaubis), dims=1)

    h3u, h3v = ftau(t, w2u + h2u, w2v + h2v, A1, A2, matr, conjmatr, sigma,
                    llambda)

    h3u = h3u - dth2u
    h3v = h3v - dth2v

    h3u_fft = fft(h3u, dims=1)
    h3v_fft = fft(h3v, dims=1)

    h3u_fft[1, :] .= 0
    h3v_fft[1, :] .= 0 

    h3u = epsilon * ifft(h3u_fft / (1im * Ktaubis), dims=1)
    h3v = epsilon * ifft(h3v_fft / (1im * Ktaubis), dims=1)

    fft_ubar = fft_u0 - h3u[1, :]
    fft_vbar = fft_v0 - h3v[1, :]

    C3u, C3v = C3(t, fft_ubar, fft_vbar, A1, A2,
                  matr, conjmatr, sigma, llambda,
                  Ktaubis, epsilon, Ntaumm)

    fft_ug = fft_u0 - C3u[1, :]
    fft_vg = fft_v0 - C3v[1, :]

    fft_ubar, fft_vbar, fft_ug, fft_vg

end
