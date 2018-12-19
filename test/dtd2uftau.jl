function dtd2uftau(t, fft_u, fft_v, fft_du, fft_dv, 
                      fft_du2, fft_dv2, 
                      A1, A2, matr, conjmatr, 
                      sigma, llambda, Ntaumm)
    

    sigma = 1 

    u     = ifft(exp(1im*t*A1) * fft_u * matr)
    v     = ifft(exp(1im*t*A1) * fft_v * matr)
    dtu   = ifft(exp(1im*t*A1) * (1im*A1) * fft_u) * matr
    dtv   = ifft(exp(1im*t*A1) * (1im*A1) * fft_v) * matr

    du    = ifft(exp(1im*t*A1) * fft_du * matr)
    dv    = ifft(exp(1im*t*A1) * fft_dv * matr)
    dtdu  = ifft(exp(1im*t*A1) * (1im*A1) * fft_du) * matr
    dtdv  = ifft(exp(1im*t*A1) * (1im*A1) * fft_dv) * matr
        
    du2   = ifft(exp(1im*t*A1) * fft_du2 * matr)
    dv2   = ifft(exp(1im*t*A1) * fft_dv2 * matr)
    dtdu2 = ifft(exp(1im*t*A1) * (1im*A1) * fft_du2) * matr
    dtdv2 = ifft(exp(1im*t*A1) * (1im*A1) * fft_dv2) * matr

    z     = (u+conjugate(v))/2
    dtz   = (dtu+conjugate(dtv))/2
    dz    = (du+conjugate(dv))/2
    dtdz  = (dtdu+conjugate(dtdv))/2
    dz2   = (du2+conjugate(dv2))/2
    dtdz2 = (dtdu2+conjugate(dtdv2))/2

    fz1 = (  2*dtdz2*conjugate(z)*dz 
           + 2*dz2*conjugate(dtz)*dz
           + 2*dz2*conjugate(z)*dtdz
           + 2*dtz*conjugate(dz2)*dz
           + 2*z*conjugate(dtdz2)*dz
           + 2*z*conjugate(dz2)*dtdz
           + 2*dtz*dz2*conjugate(dz)
           + 2*z*dtdz2*conjugate(dz)
           + 2*z*dz2*conjugate(dtdz) )

    champu1 = -1im*llambda*A2*exp(-1im*t*A1)*fft(conjmatr*fz1)
    champv1 = -1im*llambda*A2*exp(-1im*t*A1)*fft(conjmatr*conjugate(fz1))
       
    fz1 = (  2*dz2*conjugate(z)*dz
           + 2*z*conjugate(dz2)*dz
           + 2*z*dz2*conjugate(dz))

    champu2 = -1im*llambda*A2*exp(-1im*t*A1)*(-1im*A1)*fft(conjmatr*fz1)
    champv2 = -1im*llambda*A2*exp(-1im*t*A1)*(-1im*A1)*fft(conjmatr*conjugate(fz1))

    champu=champu1+champu2
    champv=champv1+champv2

    champu,champv

end
