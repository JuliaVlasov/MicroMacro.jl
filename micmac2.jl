using FFTW, LinearAlgebra

function reconstr(u, t, T, ntau)

    w   = zeros(ComplexF64, ntau)
    w  .= vcat(0:ntau÷2-1,-ntau÷2:-1)
    w  .= exp.(1im * 2π / T * w * t)
    v   = fft(u,1)

    vec(sum(v .* w, dims=1) / ntau)

end

function ftau(m, t, fft_u :: Vector{ComplexF64}, fft_v :: Vector{ComplexF64})

    u = transpose(ifft(exp.(1im * t * m.A1) .* fft_u)) .* m.matr
    v = transpose(ifft(exp.(1im * t * m.A1) .* fft_v)) .* m.matr

    z = (u .+ conj.(v)) / 2

    fz1 = abs.(z) .^ (2 * m.sigma) .* z

    u = -1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) 
    v = -1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) 

    u = transpose(u) .* fft(m.conjmatr .* fz1)
    v = transpose(v) .* fft(m.conjmatr .* conj.(fz1))

    return u, v
end

function ftau(m, t, fft_u :: Array{ComplexF64,2}, fft_v :: Array{ComplexF64,2})

    u = ifft(transpose(exp.(1im * t * m.A1)) .* fft_u,2) .* m.matr
    v = ifft(transpose(exp.(1im * t * m.A1)) .* fft_v,2) .* m.matr

    z = (u .+ conj.(v)) / 2

    fz1 = abs.(z) .^ (2 * m.sigma) .* z

    u = -1im * m.llambda .* m.A2 .* exp.(-1im * t * m.A1) 
    v = -1im * m.llambda .* m.A2 .* exp.(-1im * t * m.A1) 

    u = transpose(u) .* fft(m.conjmatr .* fz1)
    v = transpose(v) .* fft(m.conjmatr .* conj.(fz1))

    return u, v

end


function duftau(m, t, fft_u, fft_v, fft_du, fft_dv )

    sigma = 1

    u = ifft(transpose(exp.(1im * t * m.A1) .* fft_u) .* m.matr, 2)
    v = ifft(transpose(exp.(1im * t * m.A1) .* fft_v) .* m.matr, 2)

    du = ifft(transpose(exp.(1im * t * m.A1) .* fft_du) .* m.matr, 2)
    dv = ifft(transpose(exp.(1im * t * m.A1) .* fft_dv) .* m.matr, 2)

    z  = (u  .+ conj.(v)) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z).^2 .* dz .+ z.^2 .* conj.(dz)

    u = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* fz1,2)
    v = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* conj(fz1),2)

    return u, v
end


function dtftau(m, t, fft_u, fft_v)

    sigma = 1

    u  = ifft(transpose(exp.(1im * t * m.A1) .* fft_u) .* m.matr, 2)
    v  = ifft(transpose(exp.(1im * t * m.A1) .* fft_v) .* m.matr, 2)
    du = transpose(ifft(exp.(1im * t * m.A1) .* (1im * m.A1) .* fft_u)) .* m.matr
    dv = transpose(ifft(exp.(1im * t * m.A1) .* (1im * m.A1) .* fft_v)) .* m.matr

    z  = ( u .+ conj.(v)) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z).^2 .* dz .+ z.^2 .* conj.(dz)

    u1 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* fz1, 2)
    v1 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* conj.(fz1), 2)

    fz1 = abs.(z).^2 .* z
    u2 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1)) .* fft(m.conjmatr .* fz1)
    v2 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1)) .* fft(m.conjmatr .* conj.(fz1))

    u = u1 .+ u2
    v = v1 .+ v2

    return u, v
end

function init_2(m, t, fft_u0, fft_v0)

    champu, champv = ftau(m, t, fft_u0, fft_v0)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0 
    champv_fft[1, :] .= 0 

    champu = ifft(champu_fft ./ (1im * m.ktau), 1)
    champv = ifft(champv_fft ./ (1im * m.ktau), 1)

    fft_ubar = fft_u0 .- m.epsilon * champu[1, :]
    fft_vbar = fft_v0 .- m.epsilon * champv[1, :]

    C1u, C1v = C1(m, t, fft_ubar, fft_vbar)

    fft_ug = fft_u0 .- C1u[1, :]
    fft_vg = fft_v0 .- C1v[1, :]

    return fft_ubar, fft_vbar, fft_ug, fft_vg
end

function champs_2(m, t, fft_ubar, fft_vbar, fft_ug, fft_vg)

    champu, champv = ftau(m, t, fft_ubar, fft_vbar)

    champu = fft(champu, 1)
    champv = fft(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    dtauh1u = ifft(champu, 1)
    dtauh1v = ifft(champv, 1)

    champu = champu ./ (1im * m.ktau)
    champv = champv ./ (1im * m.ktau)

    champu = m.epsilon * ifft(champu, 1)
    champv = m.epsilon * ifft(champv, 1)

    champu = transpose(fft_ubar) .+ champu
    champv = transpose(fft_vbar) .+ champv

    ffu, ffv = ftau(m, t, champu .+ transpose(fft_ug), champv .+ transpose(fft_vg))

    champu, champv = ftau(m, t, champu, champv)

    champu = fft(champu, 1)
    champv = fft(champv, 1)

    champubar = champu[1, :] / m.ntau
    champvbar = champv[1, :] / m.ntau

    champu1, champv1 = duftau(m, t, fft_ubar, fft_vbar, champubar, champvbar)

    champu2, champv2 = dtftau(m, t, fft_ubar, fft_vbar)

    champu = fft(champu1 .+ champu2, 1)
    champv = fft(champv1 .+ champv2, 1)

    champu[1, :] .= 0.0 
    champv[1, :] .= 0.0  

    champu = champu ./ (1im * m.ktau)
    champv = champv ./ (1im * m.ktau)

    champu = m.epsilon * ifft(champu, 1)
    champv = m.epsilon * ifft(champv, 1)

    champu = transpose(champubar) .+ champu
    champv = transpose(champvbar) .+ champv

    champu = ffu .- dtauh1u .- champu
    champv = ffv .- dtauh1v .- champv

    champu = fft(champu, 1)
    champv = fft(champv, 1)

    champmoyu = champu[1, :] / m.ntau
    champmoyv = champv[1, :] / m.ntau

    champu[1, :] .= 0.0  
    champv[1, :] .= 0.0  

    champu = champu ./ (1im * m.ktau)
    champv = champv ./ (1im * m.ktau)

    champu = ifft(champu, 1)
    champv = ifft(champv, 1)

    return champubar, champvbar, champu, champv, champmoyu, champmoyv

end


function C1(m, t, fft_u, fft_v )

    champu, champv = ftau(m, t, fft_u, fft_v)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    champu = ifft(champu_fft ./ (1im * m.ktau), 1)
    champv = ifft(champv_fft ./ (1im * m.ktau), 1)

    C1u = transpose(fft_u) .+ m.epsilon * champu
    C1v = transpose(fft_v) .+ m.epsilon * champv

    return C1u, C1v
end

""" 
Class with initial data Relativistic Klein-Gordon equation
"""
struct DataSet

    nx        :: Int64  
    xmin      :: Float64
    xmax      :: Float64
    epsilon   :: Float64 
    kx        :: Vector{Float64}
    T         :: Float64
    Tfinal    :: Float64
    sigma     :: Int64
    llambda   :: Int64
    x         :: Array{Float64,1}
    u         :: Array{ComplexF64,1}
    v         :: Array{ComplexF64,1}
    dx        :: Float64
    
    function DataSet( xmin, xmax, nx, epsilon, Tfinal)

        kx  = zeros(Float64, nx)
        kx .= 2 * pi / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)
        T   = 2 * pi

        x   = zeros(Float64, nx)
        x  .= range(xmin, stop=xmax, length=nx+1)[1:end-1]
        dx  = (xmax - xmin) / nx

        ϕ  = zeros(ComplexF64, nx)
        γ  = zeros(ComplexF64, nx)

        ϕ  .= (1 + 1im) * cos.(x)
        γ  .= (1 - 1im) * sin.(x)

        sigma   = 1
        llambda = -1

        u = zeros(ComplexF64, nx)
        v = zeros(ComplexF64, nx)

        u .= ϕ .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(γ))
        v .= conj.(ϕ) .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(conj.(γ)))

        new(nx, xmin, xmax, epsilon, kx, T, Tfinal, sigma, llambda, x, u, v, dx)

    end

end

struct MicMac

    data      :: DataSet
    ntau      :: Int64
    ktau      :: Vector{Float64}
    matr      :: Vector{ComplexF64}
    conjmatr  :: Vector{ComplexF64}
    A1        :: Vector{Float64}
    A2        :: Vector{Float64}
    sigma     :: Int64
    llambda   :: Int64
    epsilon   :: Float64

    function MicMac( data, ntau )

        nx        = data.nx
        T         = data.T
        kx        = data.kx

        epsilon   = data.epsilon

        llambda   = data.llambda
        sigma     = data.sigma

        tau       = zeros(Float64, ntau)
        tau      .= T * collect(0:ntau-1) / ntau
        ktau      = similar(tau)
        ktau     .= 2π / T * vcat(0:ntau÷2-1,-ntau÷2:-1)
        ktau[1]   = 1.0

        matr      = zeros(ComplexF64,ntau)
        conjmatr  = zeros(ComplexF64,ntau)
        matr     .= exp.( 1im * tau)
        conjmatr .= exp.(-1im * tau)

        A1 = zeros(Float64, nx)
        A2 = zeros(Float64, nx)

        if epsilon > 0
            A1 .= (sqrt.(1 .+ epsilon * kx .^2) .- 1) / epsilon
            A2 .= (1 .+ epsilon * kx .^2) .^ (-1/2)
        else
            A1 .= 0.5 * kx .^ 2
            A2 .= 1.0
        end

        new( data, ntau, ktau, matr, conjmatr, A1 , A2, 
             sigma, llambda, epsilon)

    end

end

function run(self, dt)

    Tfinal = self.data.Tfinal

    T  = self.data.T
    kx = self.data.kx

    u  = self.data.u
    v  = self.data.v

    epsilon = self.data.epsilon

    dx   = self.data.dx
    ktau = self.ktau

    A1 = self.A1
    A2 = self.A2

    matr     = self.matr
    conjmatr = self.conjmatr
    ntau     = self.ntau

    t = 0
    iter = 0
    fft_u0 = fft(u)
    fft_v0 = fft(v)
    fft_ubar, fft_vbar, fft_ug, fft_vg = init_2(self, 0, fft_u0, fft_v0)

    println(size(fft_ug), size(fft_vg))

    while t < Tfinal

        iter = iter + 1
        dt = min(Tfinal-t, dt)

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = champs_2(self, t, fft_ubar, fft_vbar, fft_ug, fft_vg)

        fft_ubar12 = fft_ubar .+ dt / 2 * champubaru
        fft_vbar12 = fft_vbar .+ dt / 2 * champubarv

        fft_ug12 = fft_ug .+ epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, self.ntau) .- epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) .+ dt / 2 * champmoyu
        fft_vg12 = fft_vg .+ epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, self.ntau) .- epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) .+ dt / 2 * champmoyv

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = champs_2(self, t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12)

        fft_ubar = fft_ubar .+ dt * champubaru
        fft_vbar = fft_vbar .+ dt * champubarv

        fft_ug = fft_ug .+ epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, self.ntau) .- epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) .+ dt * champmoyu

        fft_vg = fft_vg .+ epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, self.ntau) .- epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) .+ dt * champmoyv

        t = t + dt

        C1u, C1v = C1(self, t, fft_ubar, fft_vbar)

        uC1eval = reconstr(C1u, t / epsilon, T, self.ntau)
        vC1eval = reconstr(C1v, t / epsilon, T, self.ntau)

        fft_u = uC1eval .+ fft_ug
        fft_v = vC1eval .+ fft_vg

    end

    fft_u = exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon) .* fft_u
    fft_v = exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon) .* fft_v

    u = ifft(fft_u)
    v = ifft(fft_v)

    return u, v

end


function reconstr_x(uv, x, xmin, xmax)

    N  = size(uv)[2]
    nx = size(x)[1]
    L  = xmax - xmin
    UA = zeros(ComplexF64, (4, nx))
    v  = fft(uv, 2) / N

    for jj in 1:N÷2
        vv = v[:, jj]
        UA .= UA .+ vv .* exp.(1im * 2 * pi / L * (jj-1) * (x' .- xmin))
    end
    
    for jj in N÷2:N
        vv = v[:, jj]
        UA = UA .+ vv .* exp.(1im * 2 * pi / L * (jj-1-N) * (x' .- xmin))
    end

    UA

end

function compute_error(u, v, data::DataSet)

    str3 = "donnee_"
    str5 = ".txt"
    
    epsilon = data.epsilon

    if (epsilon == 10       )  str4 = "10"        end
    if (epsilon == 5        )  str4 = "5"         end
    if (epsilon == 2.5      )  str4 = "2_5"       end
    if (epsilon == 1        )  str4 = "1"         end
    if (epsilon == 0.5      )  str4 = "0_5"       end
    if (epsilon == 0.2      )  str4 = "0_2"       end
    if (epsilon == 0.25     )  str4 = "0_25"      end
    if (epsilon == 0.1      )  str4 = "0_1"       end
    if (epsilon == 0.05     )  str4 = "0_05"      end
    if (epsilon == 0.025    )  str4 = "0_025"     end
    if (epsilon == 0.01     )  str4 = "0_01"      end
    if (epsilon == 0.005    )  str4 = "0_005"     end
    if (epsilon == 0.0025   )  str4 = "0_0025"    end
    if (epsilon == 0.001    )  str4 = "0_001"     end
    if (epsilon == 0.0005   )  str4 = "0_0005"    end
    if (epsilon == 0.00025  )  str4 = "0_00025"   end
    if (epsilon == 0.0001   )  str4 = "0_0001"    end
    if (epsilon == 0.00005  )  str4 = "0_00005"   end
    if (epsilon == 0.000025 )  str4 = "0_000025"  end
    if (epsilon == 0.00001  )  str4 = "0_00001"   end
    if (epsilon == 0.000005 )  str4 = "0_000005"  end
    if (epsilon == 0.0000025)  str4 = "0_0000025" end
    if (epsilon == 0.000001 )  str4 = "0_000001"  end

    ref_file = joinpath("test", "donnees_data3_128_micmac/", str3 * str4 * str5)
    
    println(ref_file)

    uv = zeros(Float64, (4, 128))

    open(ref_file) do f

        for (j,line) in enumerate(eachline(f))
            for (i, val) in enumerate( [ parse(Float64, val) for val in split(line)]) 
                uv[i, j] = val
            end
        end

    end

    nx   = data.nx
    x    = data.x
    xmin = data.xmin
    xmax = data.xmax
    dx   = data.dx
    kx   = data.kx

    uv = reconstr_x(uv, x, xmin, xmax)

    uref = uv[1, :] .+ 1im * uv[2, :]
    vref = uv[3, :] .+ 1im * uv[4, :]

    refH1 = sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(uref)))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(vref)))^2)
    
    err  = (sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(u .- uref)))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(v .- uref)))^2)) / refH1
    
    err

end

xmin    = 0
xmax    = 2 * pi
T       = 2 * pi
N       = 64
ntau    = 32
Tfinal  = 0.25
epsilon = 0.1

data = DataSet(xmin, xmax, N, epsilon, Tfinal)

dt = 2 ^ (-4) * Tfinal / 16

m = MicMac(data, ntau)
u, v = run(m, dt)

println(compute_error(u, v, data))
