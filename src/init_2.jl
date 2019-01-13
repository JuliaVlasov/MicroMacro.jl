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

