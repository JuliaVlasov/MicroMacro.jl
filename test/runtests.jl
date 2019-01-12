using Test
using MicroMacro

include("error.jl")

begin

    epsilons = [10.0^(-i) for i in 0:6]
    
    xmin     = 0
    xmax     = 2π
    T        = 2π
    size_x   = [64]
    size_tau = [64]
    Tfinal   = 0.25
    
    nb_dt  = 5 
    
    for nx in size_x, ntau in size_tau
    
        println(rpad("dt", 15),  rpad("epsilon",15), rpad("error",15))

        for (kk, epsilon) in enumerate(epsilons)
    
            data = DataSet(xmin, xmax, nx, epsilon, T, Tfinal)
    
            for hh in 1:nb_dt
    
                dt = 2.0^(-hh) * Tfinal / 16
    
                micmac = MicMac(data, ntau)
    
                u, v = solve(micmac, dt)
    
                err = compute_error(u,v,data)
                println(rpad(dt, 15), rpad(epsilon,15), rpad(err, 15))
                @test isapprox(err, 0.0, atol=1e-5)
    
            end
    
        end
    
    end

end
