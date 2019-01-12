using FFTW, LinearAlgebra, Plots
gr()

include("src/dataset.jl")
include("src/micmac.jl")

epsilons = [10.0^(-i) for i in 0:6]

xmin     = 0
xmax     = 2π
T        = 2π
size_x   = [64]
size_tau = [64]
Tfinal   = 0.25

p = plot(layout=(1,2))

nb_dt  = 5 # different values of dt
tabdt  = zeros(Float64, nb_dt)
taberr = zeros(Float64, (length(epsilons), nb_dt))

etime = @elapsed for nx in size_x, ntau in size_tau

    println(" nx                 : $nx ")
    println(" ntau               : $ntau ")

    for (kk, epsilon) in enumerate(epsilons)

        print(" $epsilon: ")

        data = DataSet(xmin, xmax, nx, epsilon, T, Tfinal)

        for hh in 1:nb_dt

            println("dt = $(2.0^(-hh)) ")

            dtmicmac = 2.0^(-hh) * data.Tfinal / 16

            solver = MicMac(data, ntau)

            u, v = run(solver, dtmicmac)

            tabdt[hh] = dtmicmac

            err = compute_error(u,v,data)
            println(" error = $err ")
            taberr[kk,hh] = err

        end

        println()

        plot!(p[1,1], tabdt, taberr[kk, :], 
              markershape=:circle,
              xaxis=:log,
              yaxis=:log, label="epsilon=$epsilon")

    end

end

xlabel!(p[1,1],"dt")
ylabel!(p[1,1],"error")
println("Elapsed time :", etime)

for j in 1:size(taberr)[2]
    plot!(p[1,2], epsilons, taberr[:, j], 
          markershape=:circle,
    xaxis=:log, yaxis=:log, label="dt=$(tabdt[j])")
end

xlabel!(p[1,2],"epsilon")
savefig("errors.png")
