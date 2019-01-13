function champs_2!(champu   :: Array{ComplexF64,2},
                   champv   :: Array{ComplexF64,2},
                   champu1  :: Array{ComplexF64,2},
                   champv1  :: Array{ComplexF64,2},
                   champu2  :: Array{ComplexF64,2},
                   champv2  :: Array{ComplexF64,2},
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

    dtauh1u = ifft(champu, 1)
    dtauh1v = ifft(champv, 1)

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    champu .*= m.epsilon 
    champv .*= m.epsilon 

    champu .+= transpose(fft_ubar)
    champv .+= transpose(fft_vbar) 

    ffu    = similar(champu)
    ffv    = similar(champv)

    ffu   .= champu .+ transpose(fft_ug)
    ffv   .= champv .+ transpose(fft_vg)

    ftau!(m, t, ffu, ffv)

    ftau!(m, t, champu, champv)

    fft!(champu, 1)
    fft!(champv, 1)

    champubar = champu[1, :] / m.ntau
    champvbar = champv[1, :] / m.ntau

    duftau!(champu, champv, m, t, fft_ubar, fft_vbar, champubar, champvbar)

    dtftau!(champu1, champv1, champu2, champv2, m, t, fft_ubar, fft_vbar)

    champu .+= champu1 .+ champu2
    champv .+= champv1 .+ champv2

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

    champmoyu = champu[1, :] / m.ntau
    champmoyv = champv[1, :] / m.ntau

    champu[1, :] .= 0.0  
    champv[1, :] .= 0.0  

    champu ./= (1im * m.ktau)
    champv ./= (1im * m.ktau)

    ifft!(champu, 1)
    ifft!(champv, 1)

    fft_ubar .= champubar
    fft_vbar .= champvbar
    fft_ug   .= champmoyu
    fft_vg   .= champmoyv

end

