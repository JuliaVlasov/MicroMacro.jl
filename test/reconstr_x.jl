function reconstr_x(u, x, xmin, xmax)

    ntau  = size(u)[1]
    nx    = size(x)[1]
    L     = xmax - xmin
    UA    = zeros(ComplexF64, (ntau,nx))
    v     = fft(u) / nx

    for j in 1:nx÷2
        UA .+= v[:, j] .* exp.(1im * 2π / L * (j-1) * (x' .- xmin))
    end

    for j in nx÷2+1:nx
        UA .+= v[:, j] .* exp.(1im * 2π / L * (j-1-nx) * (x' .- xmin))
    end

    UA

end
