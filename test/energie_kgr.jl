function energie_kgr(u, v, epsilon, k, dx, llambda, sigma)

    z = (u + conj(v)) / 2
    dtz = 1im / 2 / epsilon * ifft(sp.sqrt(1 + epsilon * k .^ 2) * fft(u - conj(v)))

    Q = -epsilon * dx * sp.real(1im * sp.sum(dtz * conj(z)))
    E = epsilon * dx * np.linalg.norm(dtz) .^ 2 \
        + dx * np.linalg.norm(ifft(k * fft(z))) .^ 2 \
        + dx * np.linalg.norm(
        z) .^ 2 / epsilon \
        - llambda / (sigma + 1) * dx * sp.sum(abs(z) .^ (2 * sigma + 2))

    return Q, E

end
