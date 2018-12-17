# 
# Klein-Gordon relativiste
# 

""" Class with initial data """
struct DataSet

    N         :: Int64  
    epsilon   :: Float64 
    k         :: Vector{Float64}
    T         :: Float64
    Tfinal    :: Float64
    Naffich   :: Int64
    sigma     :: Int64
    lambda    :: Int64
    u         :: Vector{Float64}
    v         :: Vector{Float64}
    

    function DataSet( dataset, xmin, xmax, N, epsilon, Tfinal)

        k = 2 * pi / (xmax - xmin) * vcat(0:N÷2,N÷2-N:-1)
        T = 2 * pi

        Naffich = 1000
        option_strang = 1
        mode_affich = [2, 4, 6, 8, 10, 12, 14, 16]

        dx = (xmax - xmin) / N
        x  = range(xmin, stop=xmax, length=N+1)[1:end-1]

        if dataset == 1

            # exemple Bao-Dong
            phi     = 2 / (exp.(x .^ 2) + exp.(-x .^ 2))
            gamma   = 0 * x
            sigma   = 1
            llambda = -4

        elseif dataset == 2

            # exemple Faou-Schratz 6.2
            phi   = (2 + 1im) / sqrt(5) * cos.(x)
            gamma = (1 + 1im) / sqrt(2) * sin.(x) .+ 0.5 * cos.(x)

            # exemple Faou-Schratz 6.3
            phi   = cos.(x)
            gamma = 1 / 4 * sin(x) + 0.5 * cos(x)

            sigma   = 1
            llambda = -1

        elseif dataset == 3

            phi   = (1 + 1im) * cos.(x)
            gamma = (1 - 1im) * sin.(x)

            sigma   = 1
            llambda = -1

        end

        @. u = phi - 1im * ifft((1 + epsilon * k^2) ^ (-1 / 2) * fft(gamma))
        @. v = conj(phi) - 1im * ifft((1 + epsilon * k ^ 2) ^ (-1 / 2) * fft(conj(gamma)))

        new(N, epsilon, k, T, Tfinal, Naffich, sigma, lambda, u, v)

    end

end
