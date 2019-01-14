using FFTW, LinearAlgebra

export MicMac, solve

mutable struct MicMac

    data      :: DataSet
    ntau      :: Int64
    ktau      :: Vector{Float64}
    matr      :: Array{ComplexF64,2}
    conjmatr  :: Array{ComplexF64,2}
    A1        :: Vector{Float64}
    A2        :: Vector{Float64}
    sigma     :: Int64
    llambda   :: Int64
    epsilon   :: Float64
    u         :: Array{ComplexF64,1}
    v         :: Array{ComplexF64,1}
    ut        :: Array{ComplexF64,2}
    vt        :: Array{ComplexF64,2}
    z         :: Array{ComplexF64,2}

    function MicMac( data :: DataSet, ntau :: Int64 )

        nx        = data.nx
        T         = data.T
        kx        = data.kx

        epsilon   = data.epsilon

        llambda   = data.llambda
        sigma     = data.sigma

        tau       = zeros(Float64, ntau)
        tau      .= T * collect(0:ntau-1) / ntau
        ktau      = similar(tau)
        ktau     .= 2π / T * vcat(0:ntau÷2-1,-ntau÷2:-1)
        ktau[1]   = 1.0

        matr      = zeros(ComplexF64,(1,ntau))
        conjmatr  = zeros(ComplexF64,(1,ntau))
        matr     .= transpose(exp.( 1im * tau))
        conjmatr .= transpose(exp.(-1im * tau))

        A1 = zeros(Float64, nx)
        A2 = zeros(Float64, nx)

        if epsilon > 0
            A1 .= (sqrt.(1 .+ epsilon * kx .^2) .- 1) / epsilon
            A2 .= (1 .+ epsilon * kx .^2) .^ (-1/2)
        else
            A1 .= 0.5 * kx .^ 2
            A2 .= 1.0
        end

        u  = zeros(ComplexF64,nx)
        v  = zeros(ComplexF64,nx)
        ut = zeros(ComplexF64,(nx, ntau))
        vt = zeros(ComplexF64,(nx, ntau))
        z  = zeros(ComplexF64,(nx, ntau))

        new( data, ntau, ktau, matr, conjmatr, A1 , A2, 
             sigma, llambda, epsilon, u, v, ut, vt, z)

    end

end

include("ftau.jl")
include("C1.jl")
include("dtftau.jl")
include("duftau.jl")
include("init_2.jl")
include("champs_2.jl")

function reconstr(u, t, T, ntau)

    v    = fft(u,1)
    w    = vcat(0:ntau÷2-1, -ntau÷2:-1)
    v  .*= exp.(1im * 2π / T * w * t)

    vec(sum(v, dims=1) / ntau)

end

function solve(self, dt)

    Tfinal = self.data.Tfinal

    T  = self.data.T
    kx = self.data.kx

    epsilon = self.data.epsilon

    nx   = self.data.nx
    ntau = self.ntau

    t = 0.0
    iter = 0

    fft_u = copy(self.data.u)
    fft_v = copy(self.data.v)

    fft!(fft_u,1)
    fft!(fft_v,1)
 
    fft_ubar = copy(fft_u)
    fft_vbar = copy(fft_v)

    fft_ug = copy(fft_u)
    fft_vg = copy(fft_v)

    ichampgu = zeros(ComplexF64,(ntau,nx))
    ichampgv = zeros(ComplexF64,(ntau,nx))

    init_2!(ichampgu, ichampgv, self, t, fft_ubar, fft_vbar, fft_ug, fft_vg )

    champubaru = similar(fft_ubar)
    champubarv = similar(fft_vbar)
    champmoyu  = similar(fft_ug)
    champmoyv  = similar(fft_vg)

    while t < Tfinal

        iter = iter + 1
        dt   = min(Tfinal-t, dt)
        hdt  = dt / 2

        champubaru .= fft_ubar
        champubarv .= fft_vbar
        champmoyu  .= fft_ug
        champmoyv  .= fft_vg

        champs_2!(ichampgu, ichampgv, 
                  self, t, 
                  champubaru, champubarv, 
                  champmoyu, champmoyv ) 

        champubaru .*= hdt
        champubaru .+= fft_ubar 
        champubarv .*= hdt 
        champubarv .+= fft_vbar 

        champmoyu  .*= hdt 
        champmoyu  .+= fft_ug
        champmoyu  .+= epsilon .* reconstr(ichampgu, (t+hdt) / epsilon, T, ntau)
        champmoyu  .-= epsilon .* reconstr(ichampgu,  t      / epsilon, T, ntau) 

        champmoyv  .*= hdt 
        champmoyv  .+= fft_vg 
        champmoyv  .+= epsilon .* reconstr(ichampgv, (t+hdt) / epsilon, T, ntau) 
        champmoyv  .-= epsilon .* reconstr(ichampgv,  t      / epsilon, T, ntau) 

        champs_2!(ichampgu, ichampgv, 
                  self, t + hdt, 
                  champubaru, champubarv, 
                  champmoyu, champmoyv ) 

        fft_ubar .+= dt .* champubaru
        fft_vbar .+= dt .* champubarv

        fft_ug .+= epsilon .* reconstr(ichampgu, (t + dt) / epsilon, T, ntau) 
        fft_ug .-= epsilon .* reconstr(ichampgu,  t       / epsilon, T, ntau) 
        fft_ug .+= dt .* champmoyu

        fft_vg .+= epsilon .* reconstr(ichampgv, (t + dt) / epsilon, T, ntau) 
        fft_vg .-= epsilon .* reconstr(ichampgv,  t       / epsilon, T, ntau) 
        fft_vg .+= dt .* champmoyv

        t = t + dt

        C1!(ichampgu, ichampgv, self, t, fft_ubar, fft_vbar)

        fft_u .= reconstr(ichampgu, t / epsilon, T, ntau)
        fft_v .= reconstr(ichampgv, t / epsilon, T, ntau)

        fft_u .+= fft_ug
        fft_v .+= fft_vg

    end

    fft_u .*= exp.(1im .* sqrt.(1 .+ epsilon * kx .^ 2) .* t / epsilon)
    fft_v .*= exp.(1im .* sqrt.(1 .+ epsilon * kx .^ 2) .* t / epsilon) 

    ifft!(fft_u,1)
    ifft!(fft_v,1)

    fft_u, fft_v

end


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
    uv    = zeros(Float64, (4, ndata))

    open(ref_file) do f

        for (j,line) in enumerate(eachline(f))
            for (i, val) in enumerate( [ parse(Float64, val) for val in split(line)]) 
                uv[i, j] = val
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

    ua = zeros(ComplexF64, (4, nx))
    va = fft(uv, 2) / ndata
    k  = zeros(Float64, ndata)
    k .= 2π / L * vcat(0:ndata÷2-1,-ndata÷2:-1)

    for j in 1:ndata
        vv  = va[:, j]
	    ua .= ua .+ vv .* exp.(1im * k[j] * (x'.- xmin))
    end

    uref = ua[1, :] .+ 1im * ua[2, :]
    vref = ua[3, :] .+ 1im * ua[4, :]

    refH1 = sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(vref,1),1))^2)

    err  = (sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(u .- uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(v .- vref,1),1))^2)) / refH1
    
    err

end
