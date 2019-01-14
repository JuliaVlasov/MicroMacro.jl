export compute_error

function compute_error(u, v, data::DataSet)

    str3 = "donnee_"
    str5 = ".txt"
    
    epsilon = data.epsilon

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

    ref_file = joinpath("test", "donnees_data3_128_micmac/", str3 * str4 * str5)
    
    ndata = 128
    uv    = zeros(Float64, (ndata, 4))

    open(ref_file) do f

        for (j,line) in enumerate(eachline(f))
            for (i, val) in enumerate( [ parse(Float64, val) for val in split(line)]) 
                uv[j, i] = val
            end
        end

    end

    nx   = data.nx
    xmin = data.xmin
    xmax = data.xmax
    T    = data.T
    x    = data.x
    dx   = (xmax - xmin) / nx
    L    = xmax - xmin
    kx   = zeros(Float64, nx)
    kx   = 2π / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)

    ua = zeros(ComplexF64, (nx, 4))
    va = fft(uv, 1) / ndata
    k  = zeros(Float64, ndata)
    k .= 2π / L * vcat(0:ndata÷2-1,-ndata÷2:-1)

    for j in 1:ndata
        vv  = transpose(va[j, :])
	    ua .= ua .+ vv .* exp.(1im * k[j] * (x .- xmin))
    end

    uref = ua[:, 1] .+ 1im * ua[:, 2]
    vref = ua[:, 3] .+ 1im * ua[:, 4]

    refH1 = sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(vref,1),1))^2)

    err  = (sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(u .- uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(v .- vref,1),1))^2)) / refH1
    
    err

end
