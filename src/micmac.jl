using FFTW, LinearAlgebra

export MicMac, solve

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
    z         :: Array{ComplexF64,2}
    dz        :: Array{ComplexF64,2}

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
        ut = zeros(ComplexF64,(nx, ntau))
        vt = zeros(ComplexF64,(nx, ntau))
        z  = zeros(ComplexF64,(nx, ntau))
        dz = zeros(ComplexF64,(nx, ntau))

        new( data, ntau, ktau, matr, conjmatr, A1 , A2, 
             sigma, llambda, epsilon, u, v, ut, vt, z, dz)

    end

end


function reconstr(u, t, T, ntau)

    v    = fft(u,1)
    w    = vcat(0:ntau÷2-1, -ntau÷2:-1)
    v  .*= exp.(1im * 2π / T * w * t)

    vec(sum(v, dims=1) / ntau)

end

function ftau!(champu :: Array{ComplexF64,2},
               champv :: Array{ComplexF64,2},
               m      :: MicMac, 
               t      :: Float64,
               fft_u  :: Vector{ComplexF64}, 
               fft_v  :: Vector{ComplexF64})

    m.v  .= exp.(1im * t * m.A1)
    m.v .*= fft_u
    ifft!(m.v,1)
    m.ut .= m.v .* m.conjmatr'

    m.v  .= exp.(1im * t * m.A1)
    m.v .*= fft_v
    ifft!(m.v,1)
    m.vt .= m.v .* m.conjmatr'

    m.ut = (m.ut .+ conj.(m.vt)) / 2
    m.vt = abs.(m.ut) .^ (2 * m.sigma) .* m.ut

    m.u .= -1im * m.llambda * m.A2 .* exp.(-1im * t * m.A1)

    m.ut .= m.matr' .* m.vt
    fft!(m.ut,1)
    m.ut .*= m.u

    m.vt .= m.matr' .* conj.(m.vt)
    fft!(m.vt,1)
    m.vt .*= m.u

    transpose!(champu, m.ut)
    transpose!(champv, m.vt)

end

function ftau!(m, t, fft_u :: Array{ComplexF64,2}, 
                     fft_v :: Array{ComplexF64,2})

    m.v .= exp.(1im * t * m.A1)

    transpose!(m.ut, fft_u)
    transpose!(m.vt, fft_v)

    m.ut .*= m.v 
    m.vt .*= m.v 

    ifft!(m.ut,1)
    ifft!(m.vt,1)

    m.ut .*= m.conjmatr'
    m.vt .*= m.conjmatr'

    m.ut .= (m.ut .+ conj.(m.vt)) / 2
    m.vt .= abs.(m.ut) .^ (2 * m.sigma) .* m.ut

    m.u .= -1im * m.llambda .* m.A2 .* conj.(m.v) 

    m.ut .= m.matr' .* m.vt
    m.vt .= m.matr' .* conj.(m.vt)

    fft!(m.ut,1)
    fft!(m.vt,1)

    m.ut .*= m.u
    m.vt .*= m.u 

    transpose!(fft_u, m.ut)
    transpose!(fft_v, m.vt)

end


function duftau!(
    champu :: Array{ComplexF64,2},
    champv :: Array{ComplexF64,2},
    m      :: MicMac, 
    t      :: Float64, 
    fft_u  :: Vector{ComplexF64}, 
    fft_v  :: Vector{ComplexF64}, 
    fft_du :: Vector{ComplexF64}, 
    fft_dv :: Vector{ComplexF64} )

    sigma = 1

    m.u .= exp.(1im * t * m.A1) 
    m.v .= -1im * m.llambda * m.A2 .* conj.(m.u)

    m.ut .= (m.u .* fft_u) .* m.conjmatr'
    ifft!(m.ut, 1)

    m.vt .= (m.u .* fft_v) .* m.conjmatr'
    ifft!(m.vt, 1)

    m.z  .= (m.ut  .+ conj.(m.vt)) / 2

    m.ut .= (m.u .* fft_du) .* m.conjmatr'
    ifft!(m.ut, 1)

    m.vt .= (m.u .* fft_dv) .* m.conjmatr'
    ifft!(m.vt, 1)

    m.dz .= (m.ut .+ conj.(m.vt)) / 2

    m.z = 2 * abs.(m.z).^2 .* m.dz .+ m.z.^2 .* conj.(m.dz)

    m.ut .= m.matr' .* m.z
    m.vt .= m.matr' .* conj.(m.z)

    fft!(m.ut,1)
    fft!(m.vt,1)
    
    m.ut .*= m.v 
    m.vt .*= m.v

    transpose!(champu, m.ut) 
    transpose!(champv, m.vt)

end


function dtftau!(champu1 :: Array{ComplexF64, 2},
                 champv1 :: Array{ComplexF64, 2},
                 champu2 :: Array{ComplexF64, 2},
                 champv2 :: Array{ComplexF64, 2},
                 m       :: MicMac, 
                 t       :: Float64, 
                 fft_u   :: Vector{ComplexF64}, 
                 fft_v   :: Vector{ComplexF64})

    sigma = 1

    m.u .= exp.(1im * t * m.A1) 
    m.v .= -1im * m.llambda * m.A2 .* conj.(m.u)

    m.ut .= (m.u .* fft_u) .* m.conjmatr'
    m.vt .= (m.u .* fft_v) .* m.conjmatr'

    ifft!(m.ut, 1)
    ifft!(m.vt, 1)

    m.z .= ( m.ut .+ conj.(m.vt)) / 2

    m.ut .= ifft(m.u .* (1im * m.A1) .* fft_u,1) .* m.conjmatr'
    m.vt .= ifft(m.u .* (1im * m.A1) .* fft_v,1) .* m.conjmatr'

    m.dz .= (m.ut .+ conj.(m.vt)) / 2

    m.dz .= 2 * abs.(m.z).^2 .* m.dz .+ m.z.^2 .* conj.(m.dz)

    m.ut .= m.matr' .* m.dz
    m.vt .= m.matr' .* conj.(m.dz)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= m.v
    m.vt .*= m.v 

    transpose!(champu1, m.ut)
    transpose!(champv1, m.vt)

    m.z .= abs.(m.z).^2 .* m.z

    m.ut .= m.matr' .* m.z
    m.vt .= m.matr' .* conj.(m.z)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= (m.v .* (-1im * m.A1))
    m.vt .*= (m.v .* (-1im * m.A1)) 

    transpose!(champu2, m.ut) 
    transpose!(champv2, m.vt)


end

function init_2!(champu   :: Array{ComplexF64,2},
                 champv   :: Array{ComplexF64,2},
                 m        :: MicMac,
                 t        :: Float64, 
                 fft_ubar :: Vector{ComplexF64}, 
                 fft_vbar :: Vector{ComplexF64},
                 fft_ug   :: Vector{ComplexF64}, 
                 fft_vg   :: Vector{ComplexF64})

    ftau!(champu, champv, m, t, fft_ubar, fft_vbar)

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

    C1!(champu, champv, m, t, fft_ubar, fft_vbar)

    fft_ug .-= champu[1, :]
    fft_vg .-= champv[1, :]

end

function champs_2!(champu   :: Array{ComplexF64,2},
                   champv   :: Array{ComplexF64,2},
                   champu1  :: Array{ComplexF64,2},
                   champv1  :: Array{ComplexF64,2},
                   champu2  :: Array{ComplexF64,2},
                   champv2  :: Array{ComplexF64,2},
                   m        :: MicMac, 
                   t        :: Float64, 
                   fft_ubar :: Vector{ComplexF64}, 
                   fft_vbar :: Vector{ComplexF64}, 
                   fft_ug   :: Vector{ComplexF64}, 
                   fft_vg   :: Vector{ComplexF64})

    ftau!(champu, champv, m, t, fft_ubar, fft_vbar)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    dtauh1u = ifft(champu, 1)
    dtauh1v = ifft(champv, 1)

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champu .*= m.epsilon 
    champv .*= m.epsilon 

    champu .+= transpose(fft_ubar)
    champv .+= transpose(fft_vbar) 

    ffu    = similar(champu)
    ffv    = similar(champv)

    ffu   .= champu .+ transpose(fft_ug)
    ffv   .= champv .+ transpose(fft_vg)

    ftau!(m, t, ffu, ffv)

    ftau!(m, t, champu, champv)

    fft!(champu, 1)
    fft!(champv, 1)

    champubar = champu[1, :] / m.ntau
    champvbar = champv[1, :] / m.ntau

    duftau!(champu, champv, m, t, fft_ubar, fft_vbar, champubar, champvbar)

    dtftau!(champu1, champv1, champu2, champv2, m, t, fft_ubar, fft_vbar)

    champu .+= champu1 .+ champu2
    champv .+= champv1 .+ champv2

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0 
    champv[1, :] .= 0.0  

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champu .*= m.epsilon 
    champv .*= m.epsilon 

    champu .+= transpose(champubar)
    champv .+= transpose(champvbar) 

    champu .= ffu .- dtauh1u .- champu
    champv .= ffv .- dtauh1v .- champv

    fft!(champu, 1)
    fft!(champv, 1)

    champmoyu = champu[1, :] / m.ntau
    champmoyv = champv[1, :] / m.ntau

    champu[1, :] .= 0.0  
    champv[1, :] .= 0.0  

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    fft_ubar .= champubar
    fft_vbar .= champvbar
    fft_ug   .= champmoyu
    fft_vg   .= champmoyv

end

function C1!(champu :: Array{ComplexF64,2},
             champv :: Array{ComplexF64,2},
             m      :: MicMac,
             t      :: Float64,
             fft_u  :: Vector{ComplexF64}, 
             fft_v  :: Vector{ComplexF64} )

    ftau!(champu, champv, m, t, fft_u, fft_v)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champu .*= m.epsilon 
    champv .*= m.epsilon 

    champu .+= transpose(fft_u)
    champv .+= transpose(fft_v) 

end

function solve(self, dt)

    Tfinal = self.data.Tfinal

    T  = self.data.T
    kx = self.data.kx

    epsilon = self.data.epsilon

    nx   = self.data.nx
    dx   = self.data.dx
    ktau = self.ktau

    A1 = self.A1
    A2 = self.A2

    matr     = self.matr
    conjmatr = self.conjmatr
    ntau     = self.ntau

    t = 0.0
    iter = 0

    fft_u = copy(self.data.u)
    fft_v = copy(self.data.v)

    fft!(fft_u,1)
    fft!(fft_v,1)
 
    fft_ubar = copy(fft_u)
    fft_vbar = copy(fft_v)

    fft_ug = copy(fft_u)
    fft_vg = copy(fft_v)

    ichampgu = zeros(ComplexF64,(ntau,nx))
    ichampgv = zeros(ComplexF64,(ntau,nx))

    champu1 = zeros(ComplexF64,(ntau,nx))
    champv1 = zeros(ComplexF64,(ntau,nx))
    champu2 = zeros(ComplexF64,(ntau,nx))
    champv2 = zeros(ComplexF64,(ntau,nx))

    init_2!(ichampgu, ichampgv, self, t, fft_ubar, fft_vbar, fft_ug, fft_vg )

    champubaru = similar(fft_ubar)
    champubarv = similar(fft_vbar)
    champmoyu  = similar(fft_ug)
    champmoyv  = similar(fft_vg)

    while t < Tfinal

        iter = iter + 1
        dt   = min(Tfinal-t, dt)
        hdt  = dt / 2

        champubaru .= fft_ubar
        champubarv .= fft_vbar
        champmoyu  .= fft_ug
        champmoyv  .= fft_vg

        champs_2!(ichampgu, ichampgv, 
                  champu1, champv1,
                  champu2, champv2,
                  self, t, 
                  champubaru, champubarv, champmoyu, champmoyv ) 

        champubaru .= fft_ubar .+ hdt * champubaru
        champubarv .= fft_vbar .+ hdt * champubarv

        champmoyu  .= fft_ug .+ (
             epsilon * reconstr(ichampgu, (t + hdt) / epsilon, T, ntau)
          .- epsilon * reconstr(ichampgu,  t        / epsilon, T, ntau) 
          .+ hdt * champmoyu )

        champmoyv  .= fft_vg .+ (
             epsilon * reconstr(ichampgv, (t + hdt) / epsilon, T, ntau) 
          .- epsilon * reconstr(ichampgv,  t        / epsilon, T, ntau) 
          .+ hdt * champmoyv )

        champs_2!(ichampgu, ichampgv, 
                  champu1, champv1,
                  champu2, champv2,
                  self, t + hdt, 
                  champubaru, champubarv, champmoyu, champmoyv ) 

        fft_ubar .+= dt * champubaru
        fft_vbar .+= dt * champubarv

        fft_ug .+= (
            epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, ntau) 
         .- epsilon * reconstr(ichampgu,  t       / epsilon, T, ntau) 
         .+ dt * champmoyu )

        fft_vg .+= (
            epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, ntau) 
         .- epsilon * reconstr(ichampgv,  t       / epsilon, T, ntau) 
         .+ dt * champmoyv )

        t = t + dt

        C1!(ichampgu, ichampgv, self, t, fft_ubar, fft_vbar)

        fft_u .= reconstr(ichampgu, t / epsilon, T, ntau)
        fft_v .= reconstr(ichampgv, t / epsilon, T, ntau)

        fft_u .+= fft_ug
        fft_v .+= fft_vg

    end

    fft_u .*= exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon)
    fft_v .*= exp.(1im * sqrt.(1 .+ epsilon * kx .^ 2) * t / epsilon) 

    ifft!(fft_u,1)
    ifft!(fft_v,1)

    fft_u, fft_v

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
    
    err

end
