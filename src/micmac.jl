export MicMac

struct MicMac

    data      :: DataSet
    ntau      :: Int64
    ktau      :: Vector{Float64}
    matr      :: Vector{ComplexF64}
    A1        :: Array{Float64,2}
    A2        :: Array{Float64,2}

    function MicMac( data, ntau )

        nx        = data.nx
        T         = data.T
        k         = transpose(data.k)

        epsilon   = data.epsilon

        llambda   = data.llambda
        sigma     = data.sigma

        tau       = zeros(Float64, ntau)
        tau      .= T * collect(0:ntau-1) / ntau
        ktau      = similar(tau)
        ktau     .= 2 * pi / T * vcat(0:ntau÷2-1,-ntau÷2:-1)
        ktau[1]   = 1.0

        matr      = zeros(ComplexF64,ntau)
        matr     .= exp.( 1im * tau)

        A1 = zeros(Float64, (1,nx))
        A2 = zeros(Float64, (1,nx))

        if epsilon > 0
            A1 .= (sqrt.(1 .+ epsilon * k.^2) .- 1) / epsilon
            A2 .= (1 .+ epsilon * k.^2) .^ (-1/2)
        else
            A1 .= 0.5 * k .^ 2
            A2 .= 1
        end

        new( data, ntau, ktau, matr, A1 , A2)

    end

end

function ftau(m :: MicMac, t, fft_u, fft_v)

    println("Call ftau")

    llambda  = m.data.llambda
    sigma    = m.data.sigma
    matr     = m.matr
    conjmatr = conj.(matr)

    champu = matr .* ifft(exp.(1im * t * m.A1) .* fft_u)
    champv = matr .* ifft(exp.(1im * t * m.A1) .* fft_v)

    z = ( champu .+ conj.(champv)) / 2

    fz1  = abs.(z) .^ (2*sigma) .* z

    champu = fft(conjmatr .* fz1)        
    champu .*= (-1im * llambda * m.A2 .* exp.(-1im * t * m.A1))
    champv = fft(conjmatr .* conj.(fz1)) 
    champv .*= (-1im * llambda * m.A2 .* exp.(-1im * t * m.A1))

    champu, champv

end

function C1(m :: MicMac, t, fft_u, fft_v) 

    println("Call C1")
    champu, champv = ftau(m, t, fft_u, fft_v)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    champu = ifft(champu_fft ./ (1im * m.ktau), 1)
    champv = ifft(champv_fft ./ (1im * m.ktau), 1)

    epsilon   = m.data.epsilon

    C1u = fft_u .+ epsilon * champu
    C1v = fft_v .+ epsilon * champv

    C1u, C1v

end


function duftau(m :: MicMac, t, fft_u, fft_v, fft_du, fft_dv)

    println("Call duftau")

    sigma    = 1
    llambda  = m.llambda
    matr     = m.matr
    conjmatr = conj.(matr)

    u = ifft(exp.(1im * t .* m.A1) .* fft_u .* matr)
    v = ifft(exp.(1im * t .* m.A1) .* fft_v .* matr)

    du = ifft(exp(1im * t .* m.A1) .* fft_du .* matr)
    dv = ifft(exp(1im * t .* m.A1) .* fft_dv .* matr)

    z  = (u  .+ conj.(v) ) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z) .^ 2 * dz .+ z .^ 2 .* conj.(dz)

    champu = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* fz1)
    champv = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* conj.(fz1))

    champu, champv

end


function dtftau(m :: MicMac, t, fft_u, fft_v )

    # attention ici je n'ai code' que le cas sigma=1
    println("Call dtftau")

    sigma    = 1
    llambda  = m.llambda
    matr     = m.matr
    conjmatr = conj.(matr)

    u  = ifft(exp.(1im * t .* m.A1) .* fft_u .* matr)
    v  = ifft(exp.(1im * t .* m.A1) .* fft_v .* matr)

    du = ifft(exp.(1im * t .* m.A1) * (1im .* m.A1) .* fft_u) .* matr
    dv = ifft(exp.(1im * t .* m.A1) * (1im .* m.A1) .* fft_v) .* matr

    z  = (u  + conj.(v) ) / 2
    dz = (du + conj.(dv)) / 2

    fz1 = 2 * abs.(z) .^ 2 * dz + z .^ 2 * conj.(dz)
    champu1 = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* fz1)
    champv1 = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* conj.(fz1))

    fz1 = abs.(z) .^ 2 .* z
    champu2 = -1im * llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1) .* fft(conjmatr * fz1)
    champv2 = -1im * llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1) .* fft(conjmatr * conj.(fz1))

    champu = champu1 .+ champu2
    champv = champv1 .+ champv2

    champu, champv

end

export solve

function solve(m :: MicMac, dt, ntau)

    Tfinal    = m.data.Tfinal
    nx        = m.data.nx
    T         = m.data.T
    k         = transpose(m.data.k)

    fft_u = zeros(ComplexF64,(1,nx))
    fft_v = zeros(ComplexF64,(1,nx))

    fft_u0 = similar(fft_u)
    fft_v0 = similar(fft_v)

    fft_u = collect(transpose(m.data.u))
    fft_v = collect(transpose(m.data.v))

    fft_ubar = similar(fft_u)
    fft_vbar = similar(fft_v)

    champu = zeros(ComplexF64,(ntau,nx))
    champv = zeros(ComplexF64,(ntau,nx))

    champu_fft = similar(champu)
    champv_fft = similar(champv)

    fft_ubar12 = similar(fft_u)
    fft_vbar12 = similar(fft_v)

    fft_ug = similar(fft_u)
    fft_vg = similar(fft_v)

    fft_ug12 = similar(fft_u)
    fft_vg12 = similar(fft_v)

    fft_ugbar12 = similar(fft_u)
    fft_vgbar12 = similar(fft_v)

    epsilon   = m.data.epsilon

    dx        = m.data.dx

    llambda   = m.data.llambda
    sigma     = m.data.sigma

    t    = 0
    iter = 0

    fft!(fft_u)
    fft!(fft_v)

    fft_u0 .= fft_u
    fft_v0 .= fft_v

    champu, champv = ftau(m, 0.0, fft_u0, fft_v0)

    champu_fft = fft(champu, 1)
    champv_fft = fft(champv, 1)

    champu_fft[1, :] .= 0.0
    champv_fft[1, :] .= 0.0

    champu .= ifft(champu_fft ./ (1im * m.ktau), 1)
    champv .= ifft(champv_fft ./ (1im * m.ktau), 1)

    fft_ubar .= fft_u0 .- epsilon * transpose(champu[1, :])
    fft_vbar .= fft_v0 .- epsilon * transpose(champv[1, :])

    C1u, C1v = C1(m, 0.0, fft_ubar, fft_vbar)

    fft_ug .= fft_u0 .- transpose(C1u[1, :])
    fft_vg .= fft_v0 .- transpose(C1v[1, :])

#    while t < Tfinal
#
#        iter += 1
#        dt    = min(Tfinal-t, dt)
#
#        fft_ubar12 = fft_ubar .+ dt/2 * champubaru
#        fft_vbar12 = fft_vbar .+ dt/2 * champubarv
#
#        fft_ug12  .= fft_ug 
#	    fft_ug12 .+= epsilon * reconstr(ichampgu, (t+dt/2)/epsilon, T, ntau) 
#	    fft_ug12 .-= epsilon * reconstr(ichampgu, t/epsilon, T, ntau) 
#	    fft_ug12 .+= dt/2 * champmoyu
#
#        fft_vg12  .= fft_vg 
#	    fft_vg12 .+= epsilon * reconstr(ichampgv, (t+dt/2)/epsilon, T, ntau) 
#	    fft_vg12 .-= epsilon * reconstr(ichampgv, t/epsilon, T, ntau) 
#	    fft_vg12 .+= dt/2 * champmoyv
#
#        champu, champv = ftau(m, t+dt/2, fft_ubar12, fft_vbar12 )
#
#        champu_fft = fft(champu, 1)
#        champv_fft = fft(champv, 1)
#
#        champu_fft[1, :] .= 0.0
#        champv_fft[1, :] .= 0.0
#
#        dtauh1u = ifft(champu_fft, 1)
#        dtauh1v = ifft(champv_fft, 1)
#
#        h1u = epsilon * ifft(champu_fft ./ (1im * ktau), 1)
#        h1v = epsilon * ifft(champv_fft ./ (1im * ktau), 1)
#
#        C1u = fft_ubar12 .+ h1u
#        C1v = fft_vbar12 .+ h1v
#
#        ffu, ffv = ftau(m, t+dt/2, C1u .+ fft_ug12, C1v .+ fft_vg12,)
#
#        champu, champv = ftau(m, t, C1u, C1v)
#
#        champu_fft = fft(champu, 1)
#        champv_fft = fft(champv, 1)
#
#        champubaru = champu_fft[1, :] / m.ntau
#        champubarv = champv_fft[1, :] / m.ntau
#
#        champu1, champv1 = duftau(m, t+dt/2, fft_ubar12, fft_vbar12, 
#                                  champubaru, champubarv)
#
#        champu2, champv2 = dtftau(m, t+dt/2, fft_ubar12, fft_vbar12 )
#
#        champu_fft = fft(champu1 .+ champu2, 1)
#        champv_fft = fft(champv1 .+ champv2, 1)
#
#        champu_fft[1, :] .= 0.0
#        champv_fft[1, :] .= 0.0
#
#        dtC1u = champubaru .+ epsilon * ifft(champu_fft / (1im * ktau), 1)
#        dtC1v = champubarv .+ epsilon * ifft(champv_fft / (1im * ktau), 1)
#
#        champgu = ffu .- dtauh1u .- dtC1u
#        champgv = ffv .- dtauh1v .- dtC1v
#
#        champgu_fft = fft(champgu, 1)
#        champgv_fft = fft(champgv, 1)
#
#        champmoyu = champgu_fft[1, :] / m.ntau
#        champmoyv = champgv_fft[1, :] / m.ntau
#
#        champgu_fft[1, :] .= 0.0 
#        champgv_fft[1, :] .= 0.0
#
#        ichampgu = ifft(champgu_fft ./ (1im * ktau), 1)
#        ichampgv = ifft(champgv_fft ./ (1im * ktau), 1)
#
#        fft_ubar .+= dt * champubaru
#        fft_vbar .+= dt * champubarv
#
#        fft_ug .+= epsilon * reconstr(ichampgu, (t+dt)/epsilon, T, ntau) 
#	    fft_ug .-= epsilon * reconstr(ichampgu, t/epsilon, T, ntau) 
#	    fft_ug .+= dt * champmoyu
#
#        fft_vg .+= epsilon * reconstr(ichampgv, (t+dt)/epsilon, T, ntau) 
#	    fft_vg .-= epsilon * reconstr(ichampgv, t/epsilon, T, ntau) 
#	    fft_vg .+= dt * champmoyv
#
#        t += dt
#
#        C1u, C1v = C1(m, t, fft_ubar, fft_vbar)
#
#        uC1eval = reconstr(C1u, t / epsilon, T, ntau)
#        vC1eval = reconstr(C1v, t / epsilon, T, ntau)
#
#        fft_u = uC1eval .+ fft_ug
#        fft_v = vC1eval .+ fft_vg
#
#    end

    fft_u .= exp.(1im * sqrt.(1 .+ epsilon * k.^2) * t / epsilon) .* fft_u
    fft_v .= exp.(1im * sqrt.(1 .+ epsilon * k.^2) * t / epsilon) .* fft_v

    ifft!(fft_u)
    ifft!(fft_v)

    fft_u, fft_v

end
