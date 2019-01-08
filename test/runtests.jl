using MicroMacro
#using Plots

#include("reconstr.jl")
#include("test_reconstr.jl")
#include("ftau.jl")
#include("C1.jl")
#include("init_2.jl")
#include("champs_2.jl")
#include("adjust_step.jl")
#include("micmac.jl")
include("error.jl")
#include("dtd2uftau.jl")
#include("dtduftau.jl")
#include("dtftau.jl")
#include("duftau.jl")
#include("d2tduftau.jl")
#include("d2tftau.jl")
#include("d2uftau.jl")
#include("d3tftau.jl")
#include("d3uftau.jl")
#include("C2.jl")
#include("C3.jl")
#include("micmac.jl")
#include("adjust_step.jl")
#include("champs_3.jl")
#include("champs_4.jl")
#include("databis.jl")
#include("energie_kgr.jl")
#include("ichampf.jl")
#include("init_3.jl")
#include("init_4.jl")
include("reconstr_x.jl")

dataset = 3

#epsilons = [10.0^(-i) for i in 0:6]
epsilons = [1.0]

xmin     = 0
xmax     = 2π
T        = 2π
size_x   = [64]
size_tau = [32]
Tfinal   = 0.25

#p = plot(layout=(1,2))

nb_dt  = 1 # 5 # different values of dt
tabdt  = zeros(Float64, nb_dt)
taberr = zeros(Float64, (length(epsilons), nb_dt))

etime = @elapsed for nx in size_x, ntau in size_tau

    println(" nx                 : $nx ")
    println(" ntau               : $ntau ")

    for (kk, epsilon) in enumerate(epsilons)

        print(" $epsilon: ")

        data = DataSet(dataset, xmin, xmax, nx, epsilon, Tfinal)

        for hh in 1:nb_dt

            println("dt = $(2.0^(-hh)) ")

            dtmicmac = 2.0^(-hh) * data.Tfinal / 16

            solver = MicMac(data, ntau)

            u, v = solve(solver, dtmicmac, ntau)

            tabdt[hh] = dtmicmac

            taberr[kk,hh] = erreur(u,v,epsilon,dataset)

        end

        println()

#        plot!(p[1,1], tabdt, taberr[kk, :], xaxis=:log,
#              yaxis=:log, label="ϵ=$epsilon")

    end

end

#xlabel!("dt")
#ylabel!("error")
#println("Elapsed time :", etime)
#
#for j in 1:size(taberr)[1]
#    plot!(p[1,2], tabepsilon, taberr[:, j], label="dt=$tabdt[j]")
#end
#
#xlabel!("epsilon")
