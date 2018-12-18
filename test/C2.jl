function C2(t, fft_u, fft_v, 
	    A1, A2, matr, conjmatr, 
	    sigma, llambda, Ktaubis, 
	    epsilon, Ntaumm)

    h1u, h1v = ftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda)

    h1u_fft = fft(h1u, dims=0)
    h1v_fft = fft(h1v, dims=0)

    dtu0u = h1u_fft[1, :] / Ntaumm
    dtu0v = h1v_fft[1, :] / Ntaumm

    h1u_fft[1, :] .= 0.0
    h1v_fft[1, :] .= 0.0

    h1u = epsilon * ifft(h1u_fft ./ (1im * Ktaubis), dims=1)
    h1v = epsilon * ifft(h1v_fft ./ (1im * Ktaubis), dims=1)

    champu1, champv1 = duftau(t, fft_u, fft_v, 
                              dtu0u, dtu0v, 
			      A1, A2, matr, 
			      conjmatr, sigma, 
			      llambda, Ntaumm)

    champu2, champv2 = dtftau(t, fft_u, fft_v, 
			      A1, A2, matr, 
			      conjmatr, sigma, 
			      llambda, Ntaumm)

    champu .= champu1 + champu2
    champv .= champv1 + champv2
    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)
    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 
    dth1u = epsilon * ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    dth1v = epsilon * ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    h2u, h2v = ftau(t, 
		    fft_u .+ h1u, 
		    fft_v .+ h1v,
		    A1, A2, matr, conjmatr,
                    sigma, llambda)
    h2u = h2u .- dth1u
    h2v = h2v .- dth1v
    h2u_fft = fft(h2u, dims=1)
    h2v_fft = fft(h2v, dims=1)
    h2u_fft[1, :] .= 0.0
    h2v_fft[1, :] .= 0.0
    h2u = epsilon * ifft(h2u_fft ./ (1im * Ktaubis), dims=1)
    h2v = epsilon * ifft(h2v_fft ./ (1im * Ktaubis), dims=1)

    C2u = fft_u .+ h2u
    C2v = fft_v .+ h2v

    return C2u, C2v

end
