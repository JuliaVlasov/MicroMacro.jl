function reconstr_x(u, x, xmin, xmax)

    ntau  = size(u)[1]
    nx    = size(u)[2]
    L     = xmax - xmin
    UA    = zeros(ComplexF64, ntau)
    v     = fft(u) / nx

    for j in 1:ntau÷2
        UA .+= v[:, j] .* exp.(1im * 2π / L * (j-1) * (x .- xmin))
    end

    for j in ntau÷2+1:ntau
        UA .+= v[:, j] .* exp.(1im * 2π / L * (j-1-nx) * (x .- xmin))
    end

    UA

end
