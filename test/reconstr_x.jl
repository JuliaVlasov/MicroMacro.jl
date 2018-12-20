function reconstr_x(u, x, xmin, xmax)

    N = sp.shape(u)[1]
    L = xmax - xmin
    UA = sp.zeros((sp.shape(u)[0], 1))
    v = fft(u) / N

    for jj in range(N // 2)
        vv = v[:, jj].reshape(sp.shape(v)[0], 1)
        UA = UA + vv * exp(1im * 2 * pi / L * jj * (x - xmin))
    end

    for jj in range(N // 2, N)
        vv = v[:, jj].reshape(sp.shape(v)[0], 1)
        UA = UA + vv * exp(1im * 2 * pi / L * (jj - N) * (x - xmin))
    end

    return UA

end
