function read_refdata(epsilon, dataset)

    str0 = ["donnees_cubique_128_micmac/",
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

    ref_file = joinpath("test", str0[dataset], str3 * str4 * str5)

    uv = zeros(Float64, (4, 128))

    open(ref_file) do f

        for (j,line) in enumerate(eachline(f))
            for (i, val) in enumerate( [ parse(Float64, val) for val in split(line)]) 
                uv[i, j] = val
            end
        end

    end

    nx   = size(u)[2]
    xmin = 0
    xmax = 2π
    T    = 2π
    t    = 0.25

    dx = (xmax - xmin) / nx
    x  = collect(range(xmin, stop=xmax, length=nx+1)[1:end-1])
    k  = collect(2π / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1))

    uv = reconstr_x(uv, x, xmin, xmax)

    uref = vec(uv[1, :] .+ 1im * uv[2, :])
    vref = vec(uv[3, :] .+ 1im * uv[4, :])

    uref, vref

end
