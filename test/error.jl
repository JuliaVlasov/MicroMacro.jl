using FFTW, LinearAlgebra

function linear(U, temps, T, Ntau)
    dtau   = T / Ntau
    Ubis   = vcat(U, transpose(U[1, :]))
    temps  = temps % T
    repere = temps / dtau + 1
    indice = trunc(Int64,repere)+1
    (indice + 1 - repere) * Ubis[indice, :] + (repere - indice) * Ubis[indice + 1, :]
end


function trigo(U, temps, T, Ntau)

    W  = zeros(ComplexF64, Ntau)
    for i in 0:Ntau-1
        W[i+1] = i
    end
    for i in Ntau÷2:Ntau-1
        W[i] = W[i] - Ntau
    end
    W = exp.(1im * 2 * pi / T * W * temps)
    V = fft(U, dims=1)

    sum(V * transpose(W),dims=0) / Ntau

end


function reconstr(U, temps, T, Ntau,type_reconst = 2)

    if type_reconst == 1

        linear(U, temps, T, Ntau)

    else

        trigo(U, temps, T, Ntau)

    end

end

function erreur(u, v, epsilon, dataset)

    # ici la reference a été calculee par micmac

    str0 = ["", "donnees_cubique_128_micmac/",
            "donnees_FS_128_micmac/",
            "donnees_data3_128_micmac/"]

    str3 = "donnee_"
    str5 = ".txt"

    if (epsilon == 10       )  str4 = "10"        end
    if (epsilon == 5        )  str4 = "5"         end
    if (epsilon == 2.5      )  str4 = "2_5"       end
    if (epsilon == 1        )  str4 = "1"         end
    if (epsilon == 0.5      )  str4 = "0_5"       end
    if (epsilon == 0.2      )  str4 = "0_2"       end
    if (epsilon == 0.25     )  str4 = "0_25"      end
    if (epsilon == 0.1      )  str4 = "0_1"       end
    if (epsilon == 0.05     )  str4 = "0_05"      end
    if (epsilon == 0.025    )  str4 = "0_025"     end
    if (epsilon == 0.01     )  str4 = "0_01"      end
    if (epsilon == 0.005    )  str4 = "0_005"     end
    if (epsilon == 0.0025   )  str4 = "0_0025"    end
    if (epsilon == 0.001    )  str4 = "0_001"     end
    if (epsilon == 0.0005   )  str4 = "0_0005"    end
    if (epsilon == 0.00025  )  str4 = "0_00025"   end
    if (epsilon == 0.0001   )  str4 = "0_0001"    end
    if (epsilon == 0.00005  )  str4 = "0_00005"   end
    if (epsilon == 0.000025 )  str4 = "0_000025"  end
    if (epsilon == 0.00001  )  str4 = "0_00001"   end
    if (epsilon == 0.000005 )  str4 = "0_000005"  end
    if (epsilon == 0.0000025)  str4 = "0_0000025" end
    if (epsilon == 0.000001 )  str4 = "0_000001"  end

    fichier = joinpath(str0[dataset], str3 * str4 * str5)

    uv = zeros(Float64, (4, 128))

    open(fichier) do azerty

        for j in 1:128
            uv[1, j] = parse(Float64, read(azerty, String))
            uv[2, j] = parse(Float64, read(azerty, String))
            uv[3, j] = parse(Float64, read(azerty, String))
            uv[4, j] = parse(Float64, read(azerty, String))
            dummy = read(azerty, String)
        end

    end

    if dataset == 1
        NN   = size(u)[2]
        xmin = -8
        xmax = 8
        t    = 0.4
    elseif dataset == 2
        NN   = size(u)[2]
        xmin = 0
        xmax = 2 * pi
        t = 1
    elseif dataset == 3
        NN = size(u)[2]
        xmin = 0
        xmax = 2 * pi
        T = 2 * pi
        t = 0.25
    end

    dx = (xmax - xmin) / NN
    x  = range(xmin, stop=xmax - dx, length=NN)
    dx = x[1, 2] - x[1, 1]
    k  = 2 * pi / (xmax - xmin) .* vcat(0:NN÷2-1, NN÷2-NN:0)

    if size(uv)[2] != NN
        uv = reconstr_x(uv, x, xmin, xmax)
    end

    uvu = uv[1, :] .+ 1im * uv[2, :]
    uvv = uv[3, :] .+ 1im * uv[4, :]

    @. refH1 = sqrt.(dx * norm(ifft(1im * sqrt(1 + k^2) * fft(uvu)))^2 + dx * norm(
        ifft(1im * sqrt(1 + k^2) * fft(uvv)))^2)
    @. err = (sqrt(dx * norm(ifft(1im * sqrt(1 + k^2) * fft(u - uvu)))^2 + dx * norm(
        ifft(1im * sqrt(1 + k^2) * fft(v - uvv)))^2)) / refH1

    err

end
