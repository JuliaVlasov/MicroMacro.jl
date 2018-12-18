using MicroMacro
using Plots

include("error.jl")
include("databis.jl")
include("micmac.jl")

dataset = 3

tabepsilon = [10.0^(-i) for i in 0:6]
tabschema  = [2]

xmin        = 0
xmax        = 2*pi
T           = 2*pi
tabsize_x   = [64]
tabsize_tau = [64]
Tfinal      = 0.25


p = plot(layout=(1,2))

nombre = 5 #nombre de valeurs de dt
tabdt  = zeros(Float64, nombre)
taberr = zeros(Float64, (length(tabepsilon), nombre))

etime = @elapsed for N in tabsize_x, Ntaumm in tabsize_tau, schema_micmac in tabschema
     
    numero = 0
    for (kk, epsilon) in enumerate(tabepsilon)

        print(" $epsilon: ")

        data = DataSet(dataset, xmin, xmax, N, epsilon, Tfinal)

        for hh in 1:nombre

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
