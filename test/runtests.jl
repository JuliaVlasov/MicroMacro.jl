using MicroMacro

include("error.jl")
include("databis.jl")
include("micmac.jl")

#dataset=3
#
#tabepsilon=[10**(-i) for i in range(7)]
#tabschema=[2]
#
#if dataset==1:
#    xmin=-8
#    xmax=8
#    T=2*pi
#    tabsize_x=[200]
#    tabsize_tau=[32]
#    Tfinal=0.4
#elif dataset==2:
#    xmin=0
#    xmax=2*pi
#    T=2*pi
#    tabsize_x=[64]
#    tabsize_tau=[64]
#    Tfinal=1
#elif dataset==3:
#    xmin=0
#    xmax=2*pi
#    T=2*pi
#    tabsize_x=[64]
#    tabsize_tau=[64]
#    Tfinal=0.25
#
#plt.figure(figsize=(15,5))
#plt.subplot(1, 2, 1)
#
#nombre = 5 #nombre de valeurs de dt
#tabdt = np.zeros(nombre)
#taberr = np.zeros((np.size(tabepsilon), nombre))
#
#for N in tabsize_x:
#    for Ntaumm in tabsize_tau:
#        for schema_micmac in tabschema:
#
#            numero=0
#            for kk, epsilon in enumerate(tabepsilon):
#
#                print("{0:.6f}".format(epsilon), end=":")
#
#                data = DataSet(dataset, xmin, xmax, N, epsilon, Tfinal)
#
#                for hh in range(nombre):
#                    print(hh, end=",")
#                    dtmicmac=2**(-hh)*data.Tfinal/16
#
#                    solver = MicMac(data)
#                    u, v = solver.run(dtmicmac, Ntaumm, schema_micmac)
#
#                    tabdt[hh]=dtmicmac
#
#                    taberr[kk,hh]=erreur(u,v,epsilon,dataset)
#
#                print()
#                plt.loglog(tabdt, taberr[kk, :], 's-', label='$\epsilon=$' + str(epsilon))
#
#plt.xlabel("dt")
#plt.ylabel("error")
#plt.legend()
#print("Elapsed time :", time.time() - tbegin)
#
#plt.subplot(1, 2, 2)
#
#for j in range(sp.shape(taberr)[1]):
#    plt.loglog(tabepsilon, taberr[:, j], 's-', label='dt=' + str(tabdt[j]))
#
#plt.xlabel("epsilon")
#plt.legend()
#plt.show()
#
#
##polyno(kk,:)=polyfit(log(tabdt),log(taberr(kk,:)),1)
