function ftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda)

    u = matr * transpose(ifft(exp.(1im * t * A1) .* fft_u)) 
    v = matr * transpose(ifft(exp.(1im * t * A1) .* fft_v))

    z = ( u .+ conj.(v)) / 2

    println(size(z))

    fz1  = abs.(z) .^ (2*sigma) .* z

    println(size(fz1))

    u = fft(conjmatr .* fz1)        
    u = u * (-1im * llambda * A2 .* exp.(-1im * t * A1))
    v = fft(conjmatr .* conj.(fz1)) 
    v = v * (-1im * llambda * A2 .* exp.(-1im * t * A1))

    println()
    println(u)
    u, v

end
