function d3tftau(t,fft_u,fft_v,A1,A2,matr,conjmatr,sigma,llambda,Ntaumm)

    sigma=1

    u=ifft(exp(1j*t*A1)*fft_u*matr)
    v=ifft(exp(1j*t*A1)*fft_v*matr)
    du=ifft(exp(1j*t*A1)*(1j*A1)*fft_u)*matr
    dv=ifft(exp(1j*t*A1)*(1j*A1)*fft_v)*matr
    d2u=ifft(exp(1j*t*A1)*(-A1**2)*fft_u)*matr
    d2v=ifft(exp(1j*t*A1)*(-A1**2)*fft_v)*matr
    d3u=ifft(exp(1j*t*A1)*(-1j*A1**3)*fft_u)*matr
    d3v=ifft(exp(1j*t*A1)*(-1j*A1**3)*fft_v)*matr

    z=(u+conj(v))/2
    dz=(du+conj(dv))/2
    d2z=(d2u+conj(d2v))/2
    d3z=(d3u+conj(d3v))/2

    fz1=2*dz*conj(z)*d2z+2*z*conj(dz)*d2z+2*z*conj(z)*d3z+4*d2z*dz*conj(z)+2*dz**2*conj(dz)+4*d2z*conj(dz)*z+4*dz*conj(d2z)*z+4*dz*conj(dz)*dz+2*dz*z*conj(d2z)+z**2*conj(d3z)
    champu1=-1j*llambda*A2*exp(-1j*t*A1)*fft(conjmatr*fz1)
    champv1=-1j*llambda*A2*exp(-1j*t*A1)*fft(conjmatr*conj(fz1))

    fz1=2*abs(z)**2*d2z+2*dz**2*conj(z)+4*abs(dz)**2*z+z**2*conj(d2z)
    champu2=-1j*llambda*A2*exp(-1j*t*A1)*(-1j*A1)*fft(conjmatr*fz1)
    champv2=-1j*llambda*A2*exp(-1j*t*A1)*(-1j*A1)*fft(conjmatr*conj(fz1))

    fz1=2*abs(z)**2*dz+z**2*conj(dz)
    champu3=-1j*llambda*A2*exp(-1j*t*A1)*(-A1**2)*fft(conjmatr*fz1)
    champv3=-1j*llambda*A2*exp(-1j*t*A1)*(-A1**2)*fft(conjmatr*conj(fz1))
       
    fz1=abs(z)**2*z
    champu4=-1j*llambda*A2*exp(-1j*t*A1)*(1j*A1**3)*fft(conjmatr*fz1)
    champv4=-1j*llambda*A2*exp(-1j*t*A1)*(1j*A1**3)*fft(conjmatr*conj(fz1))

    champu=champu1+3*champu2+3*champu3+champu4
    champv=champv1+3*champv2+3*champv3+champv4

    return champu,champv

end
