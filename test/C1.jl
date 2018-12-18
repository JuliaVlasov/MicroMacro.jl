function C1(t, fft_u, fft_v, A1, A2, 
	    matr, conjmatr, sigma, 
            llambda, Ktaubis, epsilon, Ntaumm)

    champu, champv = ftau(t, fft_u, fft_v, A1, A2, 
			  matr, conjmatr, sigma, llambda)

    champu_fft = fft(champu, dims=1)
    champv_fft = fft(champv, dims=1)

    n = size(champu)[2]

    for i in 1:n
        champu_fft[1, i] = 0.0
        champv_fft[1, i] = 0.0
    end

    champu = ifft(champu_fft ./ (1im * Ktaubis), dims=1)
    champv = ifft(champv_fft ./ (1im * Ktaubis), dims=1)

    C1u = fft_u .+ epsilon * champu
    C1v = fft_v .+ epsilon * champv

    C1u, C1v

end
