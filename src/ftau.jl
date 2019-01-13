function ftau!(champu :: Array{ComplexF64,2},
               champv :: Array{ComplexF64,2},
               m      :: MicMac, 
               t      :: Float64,
               fft_u  :: Vector{ComplexF64}, 
               fft_v  :: Vector{ComplexF64})

    m.v  .= exp.(1im * t * m.A1)
    m.u .= -1im * m.llambda * m.A2 .* conj.(m.v)

    m.v .*= fft_u
    ifft!(m.v,1)
    m.ut .= m.v .* m.matr

    m.v  .= exp.(1im * t * m.A1)
    m.v .*= fft_v
    ifft!(m.v,1)
    m.vt .= m.v .* m.matr

    m.ut .+= conj.(m.vt)
    m.ut .*= 0.5
    m.vt = abs.(m.ut) .^ (2 * m.sigma) .* m.ut

    m.ut .= m.conjmatr .* m.vt
    fft!(m.ut,1)
    m.ut .*= m.u

    m.vt .= m.conjmatr .* conj.(m.vt)
    fft!(m.vt,1)
    m.vt .*= m.u

    transpose!(champu, m.ut)
    transpose!(champv, m.vt)

end

function ftau!(m     :: MicMac, 
               t     :: Float64,
               fft_u :: Array{ComplexF64,2}, 
               fft_v :: Array{ComplexF64,2})

    m.v .= exp.(1im * t * m.A1)

    transpose!(m.ut, fft_u)
    transpose!(m.vt, fft_v)

    m.ut .*= m.v 
    m.vt .*= m.v 

    ifft!(m.ut,1)
    ifft!(m.vt,1)

    m.ut .*= m.matr
    m.vt .*= m.matr

    m.ut .+= conj.(m.vt)
    m.ut .*= 0.5
    m.vt .= abs.(m.ut) .^ (2 * m.sigma) .* m.ut

    m.u .= -1im * m.llambda .* m.A2 .* conj.(m.v) 

    m.ut .= m.conjmatr .* m.vt
    m.vt .= m.conjmatr .* conj.(m.vt)

    fft!(m.ut,1)
    fft!(m.vt,1)

    m.ut .*= m.u
    m.vt .*= m.u 

    transpose!(fft_u, m.ut)
    transpose!(fft_v, m.vt)

end


