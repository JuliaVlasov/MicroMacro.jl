using Test

@testset "Test of reconstr function" begin

T    = 2 * pi

Ntaumm = 64

tau  = zeros(Float64, Ntaumm)
tau .= T * collect(0:Ntaumm-1) / Ntaumm

aa = exp.(cos.(tau)) * transpose(ones(4))

bb = reconstr(aa, 1, T, Ntaumm)

@test isapprox(vec(abs.(bb .- exp(cos(1)))), zeros(length(bb)), atol=1e-15 )

end
