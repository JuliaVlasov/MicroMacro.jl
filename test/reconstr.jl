function linear(U, temps, T, Ntau)

    dtau   = T / Ntau
    Ubis   = vcat(U, U[0, :])
    temps  = temps % T
    repere = temps / dtau + 1
    indice = trunc(Int64,repere)
    return (indice + 1 - repere) * Ubis[indice, :] + (repere - indice) * Ubis[indice + 1, :]

end


function trigo(U, temps, T, Ntau)

    W = np.arange(Ntau, dtype=complex)
    W[Ntau ÷ 2:end] .-= Ntau
    W = exp.(1im * 2 * pi / T * W * temps)
    V = fft(U, dims=1)
    UA = sum(V * W,dims=1) / Ntau

    UA

end


function reconstr(U, temps, T, Ntau,type_reconst = 2)

    if type_reconst == 1

        res = linear(U, temps, T, Ntau)

    else

        res = trigo(U, temps, T, Ntau)

    end

    res

end
