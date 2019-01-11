using FFTW, LinearAlgebra

function reconstr(u, t, T, ntau)

    w   = zeros(ComplexF64, ntau)
    w  .= collect(0:ntau-1)
    w[ntau÷2:end] .-= ntau
    w  .= exp.(1im * 2π / T * w * t)
    v   = fft(u,1)

    vec(sum(v .* w, dims=1) / ntau)

end

function ftau(m, t, fft_u :: Vector{ComplexF64}, fft_v :: Vector{ComplexF64})

    m.v  .= exp.(1im * t * m.A1)
    m.v .*= fft_u
    ifft!(m.v,1)
    m.ut .= transpose(m.v) .* m.matr

    m.v  .= exp.(1im * t * m.A1)
    m.v .*= fft_v
    ifft!(m.v,1)
    m.vt .= transpose(m.v) .* m.matr

    m.ut = (m.ut .+ conj.(m.vt)) / 2
    m.vt = abs.(m.ut) .^ (2 * m.sigma) .* m.ut

    m.u .= -1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)

    m.ut .= m.conjmatr .* m.vt
    fft!(m.ut,2)
    m.ut .*= transpose(m.u)

    m.vt .= m.conjmatr .* conj.(m.vt)
    fft!(m.vt,2)
    m.vt .*= transpose(m.u) 

    m.ut, m.vt
end

function ftau(m, t, fft_u :: Array{ComplexF64,2}, fft_v :: Array{ComplexF64,2})

    m.v .= exp.(1im * t * m.A1)

    ut = transpose(m.v) .* fft_u
    vt = transpose(m.v) .* fft_v

    ifft!(ut,2)
    ifft!(vt,2)

    ut .*= m.matr
    vt .*= m.matr

    ut .= (ut .+ conj.(vt)) / 2
    vt .= abs.(ut) .^ (2 * m.sigma) .* ut

    m.u = -1im * m.llambda .* m.A2 .* conj.(m.v) 

    ut .= m.conjmatr .* vt
    vt .= m.conjmatr .* conj.(vt)

    fft!(ut,2)
    fft!(vt,2)

    ut .*= transpose(m.u)
    vt .*= transpose(m.u) 

    ut, vt

end


function duftau(m, t, 
    fft_u  :: Vector{ComplexF64}, fft_v  :: Vector{ComplexF64}, 
    fft_du :: Vector{ComplexF64}, fft_dv :: Vector{ComplexF64} )

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

    u, v

end


function dtftau(m, t, fft_u :: Vector{ComplexF64}, fft_v :: Vector{ComplexF64})

    sigma = 1

    u  = ifft(transpose(exp.(1im * t * m.A1) .* fft_u) .* m.matr, 2)
    v  = ifft(transpose(exp.(1im * t * m.A1) .* fft_v) .* m.matr, 2)

    du = transpose(ifft(exp.(1im * t * m.A1) .* (1im * m.A1) .* fft_u,1)) .* m.matr
    dv = transpose(ifft(exp.(1im * t * m.A1) .* (1im * m.A1) .* fft_v,1)) .* m.matr

    z  = ( u .+ conj.(v)) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z).^2 .* dz .+ z.^2 .* conj.(dz)

    u1 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* fz1, 2)
    v1 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)) .* fft(m.conjmatr .* conj.(fz1), 2)

    fz1 = abs.(z).^2 .* z
    u2 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1)) .* fft(m.conjmatr .* fz1, 2)
    v2 = transpose(-1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1)) .* fft(m.conjmatr .* conj.(fz1), 2)

    u = u1 .+ u2
    v = v1 .+ v2

    u, v

end

function init_2!(m, t, 
                 fft_ubar :: Vector{ComplexF64}, 
                 fft_vbar :: Vector{ComplexF64},
                 fft_ug   :: Vector{ComplexF64}, 
                 fft_vg   :: Vector{ComplexF64})

    champu, champv = ftau(m, t, fft_ubar, fft_vbar)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0 
    champv[1, :] .= 0 

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    fft_ubar .-= m.epsilon * champu[1, :]
    fft_vbar .-= m.epsilon * champv[1, :]

    champu, champv = C1(m, t, fft_ubar, fft_vbar)

    fft_ug .-= champu[1, :]
    fft_vg .-= champv[1, :]

end

function champs_2(m, t, 
    fft_ubar :: Vector{ComplexF64}, 
    fft_vbar :: Vector{ComplexF64}, 
    fft_ug   :: Vector{ComplexF64}, 
    fft_vg   :: Vector{ComplexF64})

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

    champubar, champvbar, champu, champv, champmoyu, champmoyv

end


function C1(m, t, fft_u :: Vector{ComplexF64}, fft_v :: Vector{ComplexF64} )

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
    
    function DataSet( xmin, xmax, nx, epsilon, T, Tfinal)

        kx  = zeros(Float64, nx)
        kx .= 2 * pi / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)

        x   = zeros(Float64, nx)
        x  .= range(xmin, stop=xmax, length=nx+1)[1:end-1]
        dx  = (xmax - xmin) / nx

        ϕ  = zeros(ComplexF64, nx)
        γ  = zeros(ComplexF64, nx)

        ϕ  .= (1 + 1im) .* cos.(x)
        γ  .= (1 - 1im) .* sin.(x)

        sigma   = 1
        llambda = -1

        u = zeros(ComplexF64, nx)
        v = zeros(ComplexF64, nx)

        u .= ϕ .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(γ,1), 1)
        v .= conj.(ϕ) .- 1im * ifft((1 .+ epsilon * kx.^2) .^ (-1/2) .* fft(conj.(γ),1),1)

        new(nx, xmin, xmax, epsilon, kx, T, Tfinal, sigma, llambda, x, u, v, dx)

    end

end

mutable struct MicMac

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
    u         :: Array{ComplexF64,1}
    v         :: Array{ComplexF64,1}
    ut        :: Array{ComplexF64,2}
    vt        :: Array{ComplexF64,2}

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

        u  = zeros(ComplexF64,nx)
        v  = zeros(ComplexF64,nx)
        ut = zeros(ComplexF64,(ntau,nx))
        vt = zeros(ComplexF64,(ntau,nx))

        new( data, ntau, ktau, matr, conjmatr, A1 , A2, 
             sigma, llambda, epsilon, u, v, ut, vt)

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

    t = 0.0
    iter = 0

    fft_u = copy(u)
    fft_v = copy(v)

    fft_u .= fft(u,1)
    fft_v .= fft(v,1)
 
    fft_ubar = similar(u)
    fft_vbar = similar(v)

    fft_ug = similar(u)
    fft_vg = similar(v)

    fft_ubar .= fft_u
    fft_vbar .= fft_v

    fft_ug .= fft_u
    fft_vg .= fft_v

    init_2!(self, 0.0, fft_ubar, fft_vbar, fft_ug, fft_vg )

    while t < Tfinal

        iter = iter + 1
        dt = min(Tfinal-t, dt)

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = champs_2(self, t, fft_ubar, fft_vbar, fft_ug, fft_vg)

        fft_ubar12 = fft_ubar .+ dt / 2 * champubaru
        fft_vbar12 = fft_vbar .+ dt / 2 * champubarv

        fft_ug12   = fft_ug .+ (
             epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, self.ntau)
          .- epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) 
          .+ dt / 2 * champmoyu )

        fft_vg12   = fft_vg .+ (
             epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, self.ntau) 
          .- epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) 
          .+ dt / 2 * champmoyv )

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = champs_2(self, t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12)

        fft_ubar .+= dt * champubaru
        fft_vbar .+= dt * champubarv

        fft_ug = fft_ug .+ (
            epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, self.ntau) 
         .- epsilon * reconstr(ichampgu, t / epsilon, T, self.ntau) 
         .+ dt * champmoyu )

        fft_vg = fft_vg .+ (
            epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, self.ntau) 
         .- epsilon * reconstr(ichampgv, t / epsilon, T, self.ntau) 
         .+ dt * champmoyv )

        t = t + dt

        C1u, C1v = C1(self, t, fft_ubar, fft_vbar)

        uC1eval = reconstr(C1u, t / epsilon, T, self.ntau)
        vC1eval = reconstr(C1v, t / epsilon, T, self.ntau)

        fft_u .= uC1eval .+ fft_ug
        fft_v .= vC1eval .+ fft_vg

    end

    fft_u .= exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon) .* fft_u
    fft_v .= exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon) .* fft_v

    ifft!(fft_u,1)
    ifft!(fft_v,1)

    return fft_u, fft_v

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
    
    ndata = 128
    uv    = zeros(Float64, (4, ndata))

    open(ref_file) do f

        for (j,line) in enumerate(eachline(f))
            for (i, val) in enumerate( [ parse(Float64, val) for val in split(line)]) 
                uv[i, j] = val
            end
        end

    end

    nx   = data.nx
    xmin = data.xmin
    xmax = data.xmax
    T    = data.T
    x    = data.x
    dx   = (xmax - xmin) / nx
    L    = xmax - xmin
    kx   = zeros(Float64, nx)
    kx   = 2π / (xmax - xmin) * vcat(0:nx÷2-1,-nx÷2:-1)

    ua = zeros(ComplexF64, (4, nx))
    va = fft(uv, 2) / ndata
    k  = zeros(Float64, ndata)
    k .= 2π / L * vcat(0:ndata÷2-1,-ndata÷2:-1)

    for j in 1:ndata
        vv  = va[:, j]
	    ua .= ua .+ vv .* exp.(1im * k[j] * (x'.- xmin))
    end

    uref = ua[1, :] .+ 1im * ua[2, :]
    vref = ua[3, :] .+ 1im * ua[4, :]

    refH1 = sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(vref,1),1))^2)

    err  = (sqrt(dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(u .- uref,1),1))^2 
               + dx * norm(ifft(1im * sqrt.(1 .+ kx.^2) .* fft(v .- vref,1),1))^2)) / refH1
    
    println(err)

    uref, vref

end

xmin    = 0
xmax    = 2 * pi
T       = 2 * pi
N       = 256
ntau    = 128
Tfinal  = 0.25
epsilon = 0.1

data = DataSet(xmin, xmax, N, epsilon, T, Tfinal)

dt = 2 ^ (-3) * Tfinal / 16

m = MicMac(data, ntau)
@time u, v = run(m, dt)

uref, vref = compute_error(u, v, data)
