function dtftau!(champu1 :: Array{ComplexF64, 2},
                 champv1 :: Array{ComplexF64, 2},
                 champu2 :: Array{ComplexF64, 2},
                 champv2 :: Array{ComplexF64, 2},
                 m       :: MicMac, 
                 t       :: Float64, 
                 fft_u   :: Vector{ComplexF64}, 
                 fft_v   :: Vector{ComplexF64})

    sigma = 1

    u = exp.(1im * t * m.A1) 
    v = -1im * m.llambda * m.A2 .* conj.(u)

    m.ut .= (u .* fft_u) .* m.matr
    m.vt .= (u .* fft_v) .* m.matr

    ifft!(m.ut, 1)
    ifft!(m.vt, 1)

    z = ( m.ut .+ conj.(m.vt)) / 2

    m.ut .= ifft(u .* (1im * m.A1) .* fft_u,1) .* m.matr
    m.vt .= ifft(u .* (1im * m.A1) .* fft_v,1) .* m.matr

    dz = (m.ut .+ conj.(m.vt)) / 2

    dz .= 2 * abs.(z).^2 .* dz .+ z.^2 .* conj.(dz)

    m.ut .= m.conjmatr .* dz
    m.vt .= m.conjmatr .* conj.(dz)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= v
    m.vt .*= v 

    transpose!(champu1, m.ut)
    transpose!(champv1, m.vt)

    z .= abs.(z).^2 .* z

    m.ut .= m.conjmatr .* z
    m.vt .= m.conjmatr .* conj.(z)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= (m.v .* (-1im * m.A1))
    m.vt .*= (m.v .* (-1im * m.A1)) 

    transpose!(champu2, m.ut) 
    transpose!(champv2, m.vt)


end

