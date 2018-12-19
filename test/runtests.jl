using MicroMacro
using Plots

include("error.jl")
include("databis.jl")
include("ftau.jl")
include("dtd2uftau.jl")
include("dtduftau.jl")
include("dtftau.jl")
include("duftau.jl")
include("d2tduftau.jl")
include("d2tftau.jl")
include("d2uftau.jl")
include("d3tftau.jl")
include("d3uftau.jl")
include("C1.jl")
include("C2.jl")
include("C3.jl")
include("micmac.jl")

dataset = 3

epsilons = [10.0^(-i) for i in 0:6]
schemes  = [2]

xmin     = 0
xmax     = 2*pi
T        = 2*pi
size_x   = [16]
size_tau = [8]
Tfinal   = 0.25


p = plot(layout=(1,2))

nb_dt  = 5 # different values of dt
tabdt  = zeros(Float64, nb_dt)
taberr = zeros(Float64, (length(epsilons), nb_dt))

etime = @elapsed for N in size_x, Ntaumm in size_tau, schema_micmac in schemes
     
    numero = 0
    for (kk, epsilon) in enumerate(epsilons)

        print(" $epsilon: ")

        data = DataSet(dataset, xmin, xmax, N, epsilon, Tfinal)

        for hh in 1:nb_dt

            print("$hh, ")

            dtmicmac = 2.0^(-hh) * data.Tfinal / 16

            solver = MicMac(data)

            u, v = run(solver, dtmicmac, Ntaumm, schema_micmac)

            tabdt[hh] = dtmicmac

            taberr[kk,hh] = erreur(u,v,epsilon,dataset)

        end

        println()

        plot!(p[1,1], tabdt, taberr[kk, :], xaxis=:log, 
              yaxis=:log, label="ϵ=$epsilon")

    end

end

xlabel!("dt")
ylabel!("error")
println("Elapsed time :", etime)

for j in 1:size(taberr)[1]
    plot!(p[1,2], tabepsilon, taberr[:, j], label="dt=$tabdt[j]")
end

xlabel!("epsilon")