function ftau(t, fft_u, fft_v, A1, A2, matr, conjmatr, sigma, llambda)

    u = ifft(exp.(1im * t * A1) .* fft_u) .* matr
    v = ifft(exp.(1im * t * A1) .* fft_v) .* matr

    z = ( u .+ conj.(v)) / 2

    fz1    = abs.(z).^(2*sigma) .* z
    champu = -1im * llambda * A2 * exp(-1im*t .* A1) .* fft(conjmatr .* fz1)
    champv = -1im * llambda * A2 * exp(-1im*t .* A1) .* fft(conjmatr .* conj.(fz1))

    champu, champv

end
