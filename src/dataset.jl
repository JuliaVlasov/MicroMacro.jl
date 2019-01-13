export DataSet
export CosSin
export FaouSchratz
export BaoDong

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

    function DataSet( xmin, xmax, nx, epsilon, T, Tfinal,
                      phi, gamma, sigma, llambda)
    
        kx  = zeros(Float64, nx)
        kx .= 2 * pi / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)
    
        x   = zeros(Float64, nx)
        x  .= range(xmin, stop=xmax, length=nx+1)[1:end-1]
        dx  = (xmax - xmin) / nx
    
        ϕ   = zeros(ComplexF64, nx)
        γ   = zeros(ComplexF64, nx)
    
        ϕ  .= phi(x)
        γ  .= gamma(x)
    
        u = zeros(ComplexF64, nx)
        v = zeros(ComplexF64, nx)
    
        u .= ϕ .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(γ,1), 1)
        v .= conj.(ϕ) .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(conj.(γ),1),1)
    
        new(nx, xmin, xmax, epsilon, kx, T, Tfinal, sigma, llambda, x, u, v, dx)

    end


end
    
function BaoDong( xmin, xmax, nx, epsilon, T, Tfinal)

    phi(x)   = 2 / (exp.(x .^ 2) + exp.(-x .^ 2))
    gamma(x) = 0

    sigma    = 1
    llambda  = -4

    DataSet(xmin, xmax, nx, epsilon, T, Tfinal, phi, gamma, sigma, llambda)

end

function CosSin( xmin, xmax, nx, epsilon, T, Tfinal)

    phi(x)   = (1 + 1im) .* cos.(x)
    gamma(x) = (1 - 1im) .* sin.(x)

    sigma   = 1
    llambda = -1

    DataSet(xmin, xmax, nx, epsilon, T, Tfinal, phi, gamma, sigma, llambda)

end


function FaouSchratz( xmin, xmax, nx, epsilon, T, Tfinal)

    # Faou-Schratz 6.2
    #ϕ  .= (2 + 1im) / sqrt(5) * cos.(x)
    #γ  .= (1 + 1im) / sqrt(2) * sin.(x) .+ 0.5 * cos.(x)

    # Faou-Schratz 6.3
    phi(x)   = cos.(x)
    gamma(x) = 1 / 4 * sin.(x) .+ 0.5 * cos.(x)

    sigma   = 1
    llambda = -1

    DataSet(xmin, xmax, nx, epsilon, T, Tfinal, phi, gamma, sigma, llambda)

end

