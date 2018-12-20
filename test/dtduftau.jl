
function dtduftau(t,
		  fft_u, fft_v,
		  fft_du, fft_dv,
                  A1, A2, 
		  matr, conjmatr,
		  sigma, llambda, Ntaumm)

    sigma=1

    u = ifft(exp.(1im * t .* A1) .* fft_u .* matr)
    v = ifft(exp.(1im * t .* A1) .* fft_v .* matr)

    dtu = ifft(exp.(1im * t .* A1) .* (1im*A1) .* fft_u) .* matr
    dtv = ifft(exp.(1im * t .* A1) .* (1im*A1) .* fft_v) .* matr

    du = ifft(exp.(1im * t .* A1) .* fft_du .* matr)
    dv = ifft(exp.(1im * t .* A1) .* fft_dv .* matr)

    dtdu=ifft(exp.(1im * t .* A1) * (1im*A1) .* fft_du)*matr
    dtdv=ifft(exp.(1im * t .* A1) * (1im*A1) .* fft_dv)*matr

    z    = (u    + conj(v)    ) / 2
    dtz  = (dtu  + conj(dtv)  ) / 2
    dz   = (du   + conj(dv)   ) / 2
    dtdz = (dtdu + conj(dtdv) ) / 2

    fz1  = (  2 * dtz * conj(z) * dz + 2 * z * conj(dtz) * dz
            + 2 * abs(z)^2 * dtdz + 2 * z * dtz * conj(dz) 
	    + z^2 * conj(dtdz))

    champu1 = -1im * llambda * A2 .* exp.(-1im * t .* A1) .* fft(conjmatr .* fz1)
    champv1 = -1im * llambda * A2 .* exp.(-1im * t .* A1) .* fft(conjmatr .* conj(fz1))
       
    fz1     = 2 * abs(z)^2 * dz + z^2 * conj(dz)
    champu2 = - 1im * llambda * A2 * exp.(-1im * t * A1) .* (-1im*A1) .* fft(conjmatr .* fz1)
    champv2 = - 1im * llambda * A2 * exp.(-1im * t * A1) .* (-1im*A1) .* fft(conjmatr .* conj(fz1))

    champu = champu1 .+ champu2
    champv = champv1 .+ champv2

    champu, champv

end
