using FFTW, LinearAlgebra


export DataSet

""" 
Class with initial data Relativistic Klein-Gordon equation
"""
struct DataSet

    nx        :: Int64  
    xmin      :: Float64
    xmax      :: Float64
    epsilon   :: Float64 
    kx        :: Vector{Float64}
    T         :: Float64
    Tfinal    :: Float64
    sigma     :: Int64
    llambda   :: Int64
    x         :: Array{Float64,1}
    u         :: Array{ComplexF64,1}
    v         :: Array{ComplexF64,1}
    dx        :: Float64
    
    function DataSet( xmin, xmax, nx, epsilon, T, Tfinal)

        kx  = zeros(Float64, nx)
        kx .= 2 * pi / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)

        x   = zeros(Float64, nx)
        x  .= range(xmin, stop=xmax, length=nx+1)[1:end-1]
        dx  = (xmax - xmin) / nx

        ϕ  = zeros(ComplexF64, nx)
        γ  = zeros(ComplexF64, nx)

        ϕ  .= (1 + 1im) .* cos.(x)
        γ  .= (1 - 1im) .* sin.(x)

        sigma   = 1
        llambda = -1

        u = zeros(ComplexF64, nx)
        v = zeros(ComplexF64, nx)

        u .= ϕ .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(γ,1), 1)
        v .= conj.(ϕ) .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(conj.(γ),1),1)

        new(nx, xmin, xmax, epsilon, kx, T, Tfinal, sigma, llambda, x, u, v, dx)

    end

end

