function C1(t,
	    fft_u, fft_v,
	    A1,    A2,
	    matr, conjmatr, sigma,
            llambda, ktau, epsilon, Ntaumm)


    println(size(fft_u))	
    println(size(fft_v))

    champu, champv = ftau(t, fft_u, fft_v, A1, A2, 
			  matr, conjmatr, sigma, llambda)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    champu = ifft(champu_fft ./ (1im * ktau), 1)
    champv = ifft(champv_fft ./ (1im * ktau), 1)

    C1u = fft_u .+ epsilon * champu
    C1v = fft_v .+ epsilon * champv

    C1u, C1v

end
