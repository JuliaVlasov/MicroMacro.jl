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

