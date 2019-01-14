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

    m.ut .= (m.u .* fft_u) .* m.matr
    m.vt .= (m.u .* fft_v) .* m.matr

    ifft!(m.ut, 1)
    ifft!(m.vt, 1)

    m.z .= ( m.ut .+ conj.(m.vt)) / 2

    m.ut .= ifft(m.u .* (1im * m.A1) .* fft_u,1) .* m.matr
    m.vt .= ifft(m.u .* (1im * m.A1) .* fft_v,1) .* m.matr

    m.dz .= (m.ut .+ conj.(m.vt)) / 2

    m.dz .= 2 * abs.(m.z).^2 .* m.dz .+ m.z.^2 .* conj.(m.dz)

    m.ut .= m.conjmatr .* m.dz
    m.vt .= m.conjmatr .* conj.(m.dz)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= m.v
    m.vt .*= m.v 

    transpose!(champu1, m.ut)
    transpose!(champv1, m.vt)

    m.z .= abs.(m.z).^2 .* m.z

    m.ut .= m.conjmatr .* m.z
    m.vt .= m.conjmatr .* conj.(m.z)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= (m.v .* (-1im * m.A1))
    m.vt .*= (m.v .* (-1im * m.A1)) 

    transpose!(champu2, m.ut) 
    transpose!(champv2, m.vt)


end

