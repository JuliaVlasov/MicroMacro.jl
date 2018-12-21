using FFTW

export DataSet

""" 
Class with initial data
 
Relativistic Klein-Gordon equation
 
"""
struct DataSet

    N         :: Int64  
    epsilon   :: Float64 
    k         :: Vector{Float64}
    T         :: Float64
    Tfinal    :: Float64
    sigma     :: Int64
    llambda   :: Int64
    u         :: Vector{ComplexF64}
    v         :: Vector{ComplexF64}
    dx        :: Float64
    
    function DataSet( dataset, xmin, xmax, N, epsilon, Tfinal)

        k  = zeros(Float64, N)
        k .= 2 * pi / (xmax - xmin) * vcat(0:N÷2-1,-N÷2:-1)
        T  = 2 * pi

        x   = zeros(Float64, N)
        x  .= range(xmin, stop=xmax, length=N+1)[1:end-1]
        dx  = (xmax - xmin) / N

        phi   = zeros(ComplexF64,N)
        gamma = zeros(ComplexF64,N)

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

        u = zeros(ComplexF64, N)
        v = zeros(ComplexF64, N)

        u .= phi .- 1im * ifft((1 .+ epsilon * k.^2) .^ (-1/2) .* fft(gamma))
        v .= conj.(phi) .- 1im * ifft((1 .+ epsilon * k.^2) .^ (-1/2) .* fft(conj.(gamma)))

        new(N, epsilon, k, T, Tfinal, sigma, llambda, u, v, dx)

    end

end
