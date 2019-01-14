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

    m.ut .= (m.u .* fft_u) .* m.matr
    ifft!(m.ut, 1)

    m.vt .= (m.u .* fft_v) .* m.matr
    ifft!(m.vt, 1)

    m.z  .= (m.ut  .+ conj.(m.vt)) / 2

    m.ut .= (m.u .* fft_du) .* m.matr
    ifft!(m.ut, 1)

    m.vt .= (m.u .* fft_dv) .* m.matr
    ifft!(m.vt, 1)

    m.dz .= (m.ut .+ conj.(m.vt)) / 2

    m.z = 2 * abs.(m.z).^2 .* m.dz .+ m.z.^2 .* conj.(m.dz)

    m.ut .= m.conjmatr .* m.z
    m.vt .= m.conjmatr .* conj.(m.z)

    fft!(m.ut,1)
    fft!(m.vt,1)
    
    m.ut .*= m.v 
    m.vt .*= m.v

    transpose!(champu, m.ut) 
    transpose!(champv, m.vt)

end


