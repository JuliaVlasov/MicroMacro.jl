using FFTW, LinearAlgebra, Plots
gr()

include("src/dataset.jl")
include("src/micmac.jl")

const epsilons = [10.0^(-i) for i in 0:6]
const xmin     = 0
const xmax     = 2π
const T        = 2π
const size_x   = [64]
const size_tau = [64]
const Tfinal   = 0.25
const nb_dt    = 5 # different values of dt

tabdt  = zeros(Float64, nb_dt)
taberr = zeros(Float64, (length(epsilons), nb_dt))

etime = @elapsed for nx in size_x, ntau in size_tau

    println(" nx                 : $nx ")
    println(" ntau               : $ntau ")

    for (kk, epsilon) in enumerate(epsilons)

        print(" $epsilon: ")

        data = CosSin(xmin, xmax, nx, epsilon, T, Tfinal)

        for hh in 0:nb_dt-1

            println("dt = $(2.0^(-hh)) ")

            dtmicmac = 2.0^(-hh) * data.Tfinal / 16

            solver = MicMac(data, ntau)

            u, v = solve(solver, dtmicmac)

            tabdt[hh+1] = dtmicmac

            err = compute_error(u,v,data)
            println(" error = $err ")
            taberr[kk,hh+1] = err

        end

        println()

    end

end

println("Elapsed time :", etime)

p = plot(layout=(1,2))
xlabel!(p[1,1],"dt")
ylabel!(p[1,1],"error")

for j in 1:size(taberr)[1]
    plot!(p[1,1], 
          tabdt, taberr[j, :], 
          markershape=:circle,
          xaxis=:log,
          yaxis=:log, 
          label="epsilon=$(epsilons[j])")
end

for j in 1:size(taberr)[2]
    plot!(p[1,2], 
          epsilons, taberr[:, j], 
          markershape=:circle,
          xaxis=:log, 
          yaxis=:log, 
          label="dt=$(tabdt[j])")
end

xlabel!(p[1,2],"epsilon")

p
