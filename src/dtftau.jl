function dtftau!(champu  :: Array{ComplexF64, 2},
                 champv  :: Array{ComplexF64, 2},
                 m       :: MicMac, 
                 t       :: Float64, 
                 fft_u   :: Vector{ComplexF64}, 
                 fft_v   :: Vector{ComplexF64})

    sigma = 1

    m.u .= exp.(1im .* t .* m.A1) 
    m.v .= -1im .* m.llambda .* m.A2 .* conj.(m.u)

    m.ut .= (m.u .* fft_u) .* m.matr
    m.vt .= (m.u .* fft_v) .* m.matr

    ifft!(m.ut, 1)
    ifft!(m.vt, 1)

    m.z .= ( m.ut .+ conj.(m.vt)) ./ 2

    fft_u .*= m.u .* (1im .* m.A1) 
    fft_v .*= m.u .* (1im .* m.A1) 

    ifft!(fft_u)
    ifft!(fft_v)

    m.ut .= fft_u .* m.matr
    m.vt .= fft_v .* m.matr

    m.ut .= (m.ut .+ conj.(m.vt)) ./ 2

    m.vt .= 2 .* abs.(m.z).^2 .* m.ut .+ m.z.^2 .* conj.(m.ut)

    m.ut .= m.conjmatr .* m.vt
    m.vt .= m.conjmatr .* conj.(m.vt)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= m.v
    m.vt .*= m.v 

    champu .+= transpose(m.ut)
    champv .+= transpose(m.vt)

    m.z  .= abs.(m.z).^2 .* m.z

    m.ut .= m.conjmatr .* m.z
    m.vt .= m.conjmatr .* conj.(m.z)

    fft!(m.ut, 1)
    fft!(m.vt, 1)

    m.ut .*= (m.v .* (-1im .* m.A1))
    m.vt .*= (m.v .* (-1im .* m.A1)) 

    champu .+= transpose(m.ut)
    champv .+= transpose(m.vt)

end

