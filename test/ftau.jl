function ftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda)

    u = matr .* ifft(exp.(1im * t * A1) .* fft_u)
    v = matr .* ifft(exp.(1im * t * A1) .* fft_v)

    z = ( u .+ conj.(v)) / 2

    fz1  = abs.(z) .^ (2*sigma) .* z

    u = fft(conjmatr .* fz1)        
    u .*= (-1im * llambda * A2 .* exp.(-1im * t * A1))
    v = fft(conjmatr .* conj.(fz1)) 
    v .*= (-1im * llambda * A2 .* exp.(-1im * t * A1))

    u, v

end
