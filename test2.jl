using FFTW, LinearAlgebra, Plots
using BenchmarkTools
gr()

include("src/dataset.jl")
include("src/micmac.jl")

const epsilon  = 1.0
const xmin     = 0
const xmax     = 2π
const T        = 2π
const nx       = 64
const ntau     = 32
const Tfinal   = 0.25

println(" epsilon : $nx ")
println(" nx      : $nx ")
println(" ntau    : $ntau ")

data = CosSin(xmin, xmax, nx, epsilon, T, Tfinal)

dt = 2.0^(-4) * data.Tfinal / 16

solver = MicMac(data, ntau)

@time u, v = solve(solver, dt)
err = compute_error(u,v,data)
println(" error = $err ")
