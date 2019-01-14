function champs_2!(champu   :: Array{ComplexF64,2},
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

    champu[1, :] .= 0.0
    champv[1, :] .= 0.0

    dtauh1u = copy(champu)
    dtauh1v = copy(champv)

    ifft!(dtauh1u, 1)
    ifft!(dtauh1v, 1)

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champu .*= m.epsilon 
    champv .*= m.epsilon 

    champu .+= transpose(fft_ubar)
    champv .+= transpose(fft_vbar) 

    ffu    = copy(champu)
    ffv    = copy(champv)

    ffu   .+= transpose(fft_ug)
    ffv   .+= transpose(fft_vg)

    ftau!(m, t, ffu, ffv)

    ftau!(m, t, champu, champv)

    fft!(champu, 1)
    fft!(champv, 1)

    champubar = champu[1, :] / m.ntau
    champvbar = champv[1, :] / m.ntau

    duftau!(champu, champv, m, t, fft_ubar, fft_vbar, champubar, champvbar)

    dtftau!(champu, champv, m, t, fft_ubar, fft_vbar)

    fft_ubar .= champubar
    fft_vbar .= champvbar


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

    fft_ug .= champu[1, :] ./ m.ntau
    fft_vg .= champv[1, :] ./ m.ntau

    champu[1, :] .= 0.0  
    champv[1, :] .= 0.0  

    champu ./= (1im .* m.ktau)
    champv ./= (1im .* m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)


end

