using FFTW

function linear(U, temps, T, Ntau)

    dtau   = T / Ntau
    Ubis   = vcat(U, transpose(U[1, :]))
    temps  = temps % T
    repere = temps / dtau + 1
    indice = trunc(Int64,repere)
    
    @assert (1 <= indice <= Ntau)

    (indice + 1 - repere) * Ubis[indice, :] .+ (repere - indice) * Ubis[indice + 1, :]

end


function trigo(U, temps, T, Ntau)

    W   = zeros(ComplexF64,Ntau)
    W  .= vcat(0:Ntau÷2-1,-Ntau÷2:-1)
    W   = exp.(1im * 2 * pi / T * W * temps)
    V   = fft(U, 1)

    sum(V .* W, dims=1) / Ntau

end


function reconstr(U, temps, T, Ntau, type = 2)

    if type == 1

        res = linear(U, temps, T, Ntau)

    else

        res = trigo(U, temps, T, Ntau)

    end

    res

end
