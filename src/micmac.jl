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
            A2 .= 1.0
        end

        new( data, ntau, ktau, matr, A1 , A2)

    end

end

function ftau(m :: MicMac, t, fft_u, fft_v)

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

function ftau!(champu, champv, m :: MicMac, t, fft_u, fft_v)

    llambda  = m.data.llambda
    sigma    = m.data.sigma
    matr     = m.matr
    conjmatr = conj.(matr)

    champu .= matr .* ifft(exp.(1im * t * m.A1) .* fft_u)
    champv .= matr .* ifft(exp.(1im * t * m.A1) .* fft_v)

    z = ( champu .+ conj.(champv)) / 2

    fz1  = abs.(z) .^ (2*sigma) .* z

    champu .= fft(conjmatr .* fz1)        
    champu .*= (-1im * llambda * m.A2 .* exp.(-1im * t * m.A1))
    champv .= fft(conjmatr .* conj.(fz1)) 
    champv .*= (-1im * llambda * m.A2 .* exp.(-1im * t * m.A1))

end

function C1!(champu, champv, m :: MicMac, t, fft_u, fft_v) 

    ftau!(champu, champv, m, t, fft_u, fft_v)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    epsilon   = m.data.epsilon

    champu .= fft_u .+ epsilon * champu
    champv .= fft_v .+ epsilon * champv

end


function duftau(m :: MicMac, t, fft_u, fft_v, fft_du, fft_dv)

    sigma    = 1
    llambda  = m.data.llambda
    matr     = m.matr
    conjmatr = conj.(matr)

    u = ifft(exp.(1im * t .* m.A1) .* fft_u .* matr)
    v = ifft(exp.(1im * t .* m.A1) .* fft_v .* matr)

    du = ifft(exp.(1im * t .* m.A1) .* fft_du .* matr )
    dv = ifft(exp.(1im * t .* m.A1) .* fft_dv .* matr )

    z  = (u  .+ conj.(v) ) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z) .^ 2 .* dz .+ z .^ 2 .* conj.(dz)

    champu = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* fz1)
    champv = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* conj.(fz1))

    champu, champv

end


function dtftau(m :: MicMac, t, fft_u, fft_v )

    sigma    = 1
    llambda  = m.data.llambda
    matr     = m.matr
    conjmatr = conj.(matr)

    u  = ifft(exp.(1im * t .* m.A1) .* fft_u .* matr)
    v  = ifft(exp.(1im * t .* m.A1) .* fft_v .* matr)

    du = ifft(exp.(1im * t .* m.A1) .* (1im .* m.A1) .* fft_u) .* matr
    dv = ifft(exp.(1im * t .* m.A1) .* (1im .* m.A1) .* fft_v) .* matr

    z  = (u  .+ conj.(v) ) / 2
    dz = (du .+ conj.(dv)) / 2

    fz1 = 2 * abs.(z) .^ 2 .* dz .+ z .^ 2 .* conj.(dz)
    champu1 = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* fz1)
    champv1 = -1im * llambda * m.A2 .* exp.(-1im * t .* m.A1) .* fft(conjmatr .* conj.(fz1))

    fz1 = abs.(z) .^ 2 .* z
    champu2 = -1im * llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1) .* fft(conjmatr .* fz1)
    champv2 = -1im * llambda * m.A2 .* exp.(-1im * t * m.A1) .* (-1im * m.A1) .* fft(conjmatr .* conj.(fz1))

    champu1 .+ champv1, champu2 .+ champv2

end


function champs_2(champu, champv, m :: MicMac, t, fft_ubar, fft_vbar, fft_ug, fft_vg)

    sigma    = 1
    llambda  = m.data.llambda
    epsilon  = m.data.epsilon
    matr     = m.matr
    conjmatr = conj.(matr)

    ftau!(champu, champv, m, t, fft_ubar, fft_vbar)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    dtauh1u = ifft(champu, 1)
    dtauh1v = ifft(champv, 1)

    h1u = epsilon * ifft(champu ./ (1im * m.ktau), 1)
    h1v = epsilon * ifft(champv ./ (1im * m.ktau), 1)

    C1u = fft_ubar .+ h1u
    C1v = fft_vbar .+ h1v

    ffu, ffv = ftau(m, t, C1u .+ fft_ug, C1v .+ fft_vg)

    ftau!(champu, champv, m, t, C1u, C1v)

    fft!(champu, 1)
    fft!(champv, 1)

    champubaru = transpose(champu[1, :] / m.ntau)
    champubarv = transpose(champv[1, :] / m.ntau)

    champu1, champv1 = duftau(m, t, fft_ubar, fft_vbar, champubaru, champubarv)

    champu2, champv2 = dtftau(m, t, fft_ubar, fft_vbar)

    champu .= fft(champu1 .+ champu2, 1)
    champv .= fft(champv1 .+ champv2, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    champu = champubaru .+ epsilon * ifft(champu ./ (1im * m.ktau), 1)
    champv = champubarv .+ epsilon * ifft(champv ./ (1im * m.ktau), 1)

    champu .= ffu .- dtauh1u .- champu
    champv .= ffv .- dtauh1v .- champv

    fft!(champu, 1)
    fft!(champv, 1)

    champmoyu = transpose(champu[1, :] / m.ntau)
    champmoyv = transpose(champv[1, :] / m.ntau)

    champu[1, :] .= 0.0 
    champv[1, :] .= 0.0

    champu .= champu ./ (1im * m.ktau)
    champv .= champv ./ (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champubaru, champubarv, champu, champv, champmoyu, champmoyv

end


function reconstr(u, t, T, ntau)

    w   = zeros(ComplexF64, ntau)
    w  .= vcat(0:ntau÷2-1,-ntau÷2:-1)
    w  .= exp.(1im * 2π / T * w * t)
    v   = fft(u, 1)

    sum(v .* w, dims=1) / ntau

end

export solve

function solve(m :: MicMac, dt)

    Tfinal    = m.data.Tfinal
    nx        = m.data.nx
    ntau      = m.ntau
    T         = m.data.T
    k         = transpose(m.data.k)

    fft_u = zeros(ComplexF64,(1,nx))
    fft_v = zeros(ComplexF64,(1,nx))

    fft_u .= transpose(m.data.u)
    fft_v .= transpose(m.data.v)

    fft_u0      = similar(fft_u)
    fft_v0      = similar(fft_v)
    fft_ubar    = similar(fft_u)
    fft_vbar    = similar(fft_v)
    fft_ubar12  = similar(fft_u)
    fft_vbar12  = similar(fft_v)

    fft_ug      = similar(fft_u)
    fft_vg      = similar(fft_v)
    fft_ug12    = similar(fft_u)
    fft_vg12    = similar(fft_v)
    fft_ugbar12 = similar(fft_u)
    fft_vgbar12 = similar(fft_v)

    champu = zeros(ComplexF64,(ntau,nx))
    champv = zeros(ComplexF64,(ntau,nx))

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

    ftau!(champu, champv, m, 0.0, fft_u0, fft_v0)

    fft!(champu, 1)
    fft!(champv, 1)

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    fft_ubar .= fft_u0 .- epsilon * transpose(champu[1, :])
    fft_vbar .= fft_v0 .- epsilon * transpose(champv[1, :])

    C1!(champu, champv, m, 0.0, fft_ubar, fft_vbar)

    fft_ug .= fft_u0 .- transpose(champu[1, :])
    fft_vg .= fft_v0 .- transpose(champv[1, :])

    while t < Tfinal

        iter += 1
        dt    = min(Tfinal-t, dt)

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = ( 
            champs_2(champu, champv, m, t, fft_ubar, fft_vbar, fft_ug, fft_vg) )

        fft_ubar12 .= fft_ubar .+ dt/2 * champubaru
        fft_vbar12 .= fft_vbar .+ dt/2 * champubarv

        fft_ug12  .= fft_ug 
	    fft_ug12 .+= epsilon * reconstr(ichampgu, (t+dt/2)/epsilon, T, ntau) 
        fft_ug12 .-= epsilon * reconstr(ichampgu, t/epsilon, T, ntau) 
        fft_ug12 .+= dt/2 * champmoyu

        fft_vg12  .= fft_vg 
        fft_vg12 .+= epsilon * reconstr(ichampgv, (t+dt/2)/epsilon, T, ntau) 
        fft_vg12 .-= epsilon * reconstr(ichampgv, t/epsilon, T, ntau) 
        fft_vg12 .+= dt/2 * champmoyv

        champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = (
            champs_2(champu, champv, m, t+dt/2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12))

        fft_ubar .+= dt * champubaru
        fft_vbar .+= dt * champubarv
        
        fft_ug .+= epsilon * reconstr(ichampgu, (t+dt)/epsilon, T, ntau) 
        fft_ug .-= epsilon * reconstr(ichampgu, t/epsilon, T, ntau) 
        fft_ug .+= dt * champmoyu
        
        fft_vg .+= epsilon * reconstr(ichampgv, (t+dt)/epsilon, T, ntau) 
        fft_vg .-= epsilon * reconstr(ichampgv, t/epsilon, T, ntau) 
        fft_vg .+= dt * champmoyv
        
        t += dt
        
        C1!(champu, champv, m, t, fft_ubar, fft_vbar)
        
        uC1eval = reconstr(champu, t / epsilon, T, ntau)
        vC1eval = reconstr(champv, t / epsilon, T, ntau)
        
        fft_u = uC1eval .+ fft_ug
        fft_v = vC1eval .+ fft_vg

    end

    fft_u .= exp.(1im * sqrt.(1 .+ epsilon * k.^2) * t / epsilon) .* fft_u
    fft_v .= exp.(1im * sqrt.(1 .+ epsilon * k.^2) * t / epsilon) .* fft_v

    ifft!(fft_u)
    ifft!(fft_v)

    fft_u, fft_v

end
