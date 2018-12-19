function linear(U, temps, T, Ntau)

    dtau = T / Ntau
    Ubis = np.concatenate((U, np.array(U[0, :], ndmin=2)), axis=0)
    temps = temps % T
    repere = temps / dtau + 1
    indice = int(repere)
    return (indice + 1 - repere) * Ubis[indice, :] + (repere - indice) * Ubis[indice + 1, :]

end


function trigo(U, temps, T, Ntau)

    W = np.arange(Ntau, dtype=complex)
    W[Ntau // 2:] -= Ntau
    W = np.exp(1j * 2 * np.pi / T * W * temps)
    V = fft(U, axis=0)
    #UA = np.zeros((1, V.shape[1]),dtype=complex)
    UA = np.sum(V * W[:,np.newaxis],axis=0) / Ntau

    return UA

end


function reconstr(U, temps, T, Ntau,type_reconst = 2):

    if type_reconst == 1

        res = linear(U, temps, T, Ntau)

    else

        res = trigo(U, temps, T, Ntau)

    end

    return res

end
