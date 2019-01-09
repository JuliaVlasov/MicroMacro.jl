using FFTW

export DataSet

""" 
Class with initial data
 
Relativistic Klein-Gordon equation
 
"""

struct DataSet

    nx        :: Int64  
    epsilon   :: Float64 
    k         :: Vector{Float64}
    T         :: Float64
    Tfinal    :: Float64
    sigma     :: Int64
    llambda   :: Int64
    x         :: Array{Float64,1}
    u         :: Array{ComplexF64,1}
    v         :: Array{ComplexF64,1}
    dx        :: Float64
    
    function DataSet( dataset, xmin, xmax, nx, epsilon, Tfinal)

        k  = zeros(Float64, nx)
        k .= 2 * pi / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)
        T  = 2 * pi

        x   = zeros(Float64, nx)
        x  .= range(xmin, stop=xmax, length=nx+1)[1:end-1]
        dx  = (xmax - xmin) / nx

        phi   = zeros(ComplexF64, nx)
        gamma = zeros(ComplexF64, nx)

        if dataset == 1

            # example Bao-Dong
            phi     .= 2 / (exp.(x .^ 2) .+ exp.(-x .^ 2))
            gamma   .= 0.0 
            sigma    = 1
            llambda  = -4

        elseif dataset == 2

            # example Faou-Schratz 6.2
            phi   .= (2 + 1im) / sqrt(5) * cos.(x)
            gamma .= (1 + 1im) / sqrt(2) * sin.(x) .+ 0.5 * cos.(x)

            # example Faou-Schratz 6.3
            phi   .= cos.(x)
            gamma .= 1 / 4 * sin(x) .+ 0.5 * cos(x)

            sigma   = 1
            llambda = -1

        elseif dataset == 3

            phi   .= (1 + 1im) * cos.(x)
            gamma .= (1 - 1im) * sin.(x)

            sigma   = 1
            llambda = -1

        end

        u = zeros(ComplexF64, nx)
        v = zeros(ComplexF64, nx)

        u .= phi .- 1im * ifft((1 .+ epsilon * k.^2) .^ (-1/2) .* fft(gamma))
        v .= conj.(phi) .- 1im * ifft((1 .+ epsilon * k.^2) .^ (-1/2) .* fft(conj.(gamma)))

        new(nx, epsilon, k, T, Tfinal, sigma, llambda, x, u, v, dx)

    end

end
