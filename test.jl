include("src/dataset.jl")
include("src/micmac.jl")
include("src/error.jl")

dataset  = 3
epsilon  = 0.1
xmin     = 0
xmax     = 2π
T        = 2π
nx       = 64
ntau     = 32
tfinal   = 0.25

data = DataSet(dataset, xmin, xmax, nx, epsilon, tfinal)

dt = 2.0^(-3) * tfinal / 16

m = MicMac(data, ntau)

@time u, v = solve(m, dt)

println(compute_error(u, v, epsilon, dataset))
