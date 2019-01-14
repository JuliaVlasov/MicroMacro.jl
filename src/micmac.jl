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


