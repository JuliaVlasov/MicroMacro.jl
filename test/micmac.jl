struct MicMac

    data :: DataSet

end 


function run(micmac :: MicMac, dt, Ntaumm, schema_micmac)

    Tfinal   = micmac.data.Tfinal
    Naffich  = micmac.data.Naffich
    N        = micmac.data.N
    T        = micmac.data.T
    k        = micmac.data.k

    u        = micmac.data.u
    v        = micmac.data.v

    epsilon  = micmac.data.epsilon

    dx       = micmac.data.dx

    Ninter   = max(1, Tfinal ÷ dt ÷ Naffich)
    tau      = T * 0:Ntaumm / Ntaumm
    tau      = tau[:end-1]
    ktau     = 2 * pi / T * vcat(0:Ntaumm÷2-1,Ntaumm÷2-Ntaumm:-1)
    ktaubis  = transpose(ktau)
    ktaubis[1] = 1
    Ktaubis  = ktaubis * ones(Float64,N)
    matr     = exp( 1im * tau) * transpose(ones(Float64,N))
    conjmatr = exp(-1im * tau) * transpose(ones(Float64,N))

    if epsilon > 0
        A1 = (sqrt(1 + epsilon * k^2) - 1) / epsilon
        A2 = (1 + epsilon * k^2) ^ (-1/2)
    else
        A1 = 0.5 * k ^ 2
        A2 = 0 * k + 1
    end

    A1 = tile(A1, (Ntaumm, 1))
    A2 = tile(A2, (Ntaumm, 1))

    t    = 0
    iter = 0
    tabt = [t]

    if schema_micmac == 0

        fft_u = fft(u)
        fft_v = fft(v)

    elseif schema_micmac == 1

        ubar = u
        vbar = v
        ug = 0 * u
        vg = 0 * v
        fft_ubar = fft(ubar)
        fft_vbar = fft(vbar)
        fft_ug = fft(ug)
        fft_vg = fft(vg)
        
    elseif (schema_micmac == 2 || schema_micmac == 25)

        fft_u0 = fft(u)
        fft_v0 = fft(v)
        fft_ubar, fft_vbar, fft_ug, fft_vg = init_2(0, fft_u0, fft_v0, A1, A2,
                                                    matr, conjmatr,
                                                    micmac.data.sigma,
                                                    micmac.data.llambda,
                                                    Ktaubis, epsilon, Ntaumm)

    elseif (schema_micmac == 3 || schema_micmac == 35)

        fft_u0 = fft(u)
        fft_v0 = fft(v)
        fft_ubar, fft_vbar, fft_ug, fft_vg = init_3(0, fft_u0, fft_v0, A1, A2,
                                                    matr, conjmatr,
                                                    micmac.data.sigma,
                                                    micmac.data.llambda,
                                                    Ktaubis, epsilon, Ntaumm)

    elseif (schema_micmac == 4)

        fft_u0 = fft(u)
        fft_v0 = fft(v)

        fft_ubar, fft_vbar, fft_ug, fft_vg = init_4(0, fft_u0, fft_v0, A1, A2,
                                                    matr, conjmatr,
                                                    micmac.data.sigma,
                                                    micmac.data.llambda, 
                                                    Ktaubis, epsilon, Ntaumm)
    end

    while t < Tfinal

        iter = iter + 1
        dt, lstat = adjust_step(Tfinal, t, dt)

        if schema_micmac == 0

            champu, champv, champumoy, champvmoy = ichampf(t, fft_u, fft_v, A1, A2,
                                                           Ntaumm, matr, conjmatr,
                                                           Ktaubis, micmac.data.llambda,
                                                           micmac.data.sigma)
            fft_u = fft_u + epsilon * reconstr(champu, (t + dt) / epsilon, T, Ntaumm) \
                    - epsilon * reconstr(champu, t / epsilon, T, Ntaumm) + dt * champumoy
            fft_v = fft_v + epsilon * reconstr(champv, (t + dt) / epsilon, T, Ntaumm) \
                    - epsilon * reconstr(champv, t / epsilon, T, Ntaumm) + dt * champvmoy
            t = t + dt

        elseif schema_micmac == 2

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_2(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)
            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_2(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar = fft_ubar + dt * champubaru
            fft_vbar = fft_vbar + dt * champubarv
            fft_ug = fft_ug + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                     - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt * champmoyu
            fft_vg = fft_vg + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                     - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt * champmoyv

            t = t + dt
            C1u, C1v = C1(t, fft_ubar, fft_vbar, A1, A2, matr, conjmatr,
                          micmac.data.sigma, micmac.data.llambda,
                          Ktaubis, epsilon, Ntaumm)
            uC1eval = reconstr(C1u, t / epsilon, T, Ntaumm)
            vC1eval = reconstr(C1v, t / epsilon, T, Ntaumm)
            fft_u = uC1eval + fft_ug
            fft_v = vC1eval + fft_vg

        elseif schema_micmac == 25

            fft_ubartmp = fft_ubar
            fft_vbartmp = fft_vbar
            fft_ugtmp   = fft_ug
            fft_vgtmp   = fft_vg
            dw          = 1
            iter2       = 0

            while (dw > 1.e-12 && iter2 < 100)

                iter2        = iter2 + 1
                fft_ubartmp2 = fft_ubartmp
                fft_vbartmp2 = fft_vbartmp
                fft_ugtmp2   = fft_ugtmp
                fft_vgtmp2   = fft_vgtmp

                champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                    champs_2(t + dt / 2, (fft_ubar + fft_ubartmp) / 2,
                             (fft_vbar + fft_vbartmp) / 2,
                             (fft_ug + fft_ugtmp) / 2,
                             (fft_vg + fft_vgtmp) / 2, A1, A2, matr, conjmatr,
                             micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)
                fft_ubartmp = fft_ubar + dt * champubaru
                fft_vbartmp = fft_vbar + dt * champubarv
                fft_ugtmp = fft_ug + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                            - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt * champmoyu
                fft_vgtmp = fft_vg + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                            - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt * champmoyv
                dw = linalg.norm(fft_ubartmp - fft_ubartmp2) \
                     + linalg.norm(fft_vbartmp - fft_vbartmp2) \
                     + linalg.norm(fft_ugtmp2 - fft_ugtmp) \
                     + linalg.norm(fft_vgtmp2 - fft_vgtmp)
            end

            fft_ubar = fft_ubartmp
            fft_vbar = fft_vbartmp
            fft_ug = fft_ugtmp
            fft_vg = fft_vgtmp
            t = t + dt

            C1u, C1v = C1(t, fft_ubar, fft_vbar, A1, A2, matr, conjmatr, micmac.data.sigma,
                          micmac.data.llambda, Ktaubis, epsilon, Ntaumm)
            uC1eval = reconstr(C1u, t / epsilon, T, Ntaumm)
            vC1eval = reconstr(C1v, t / epsilon, T, Ntaumm)
            fft_u = uC1eval + fft_ug
            fft_v = vC1eval + fft_vg

        elseif schema_micmac == 3

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar14 = fft_ubar + dt / 4 * champubaru
            fft_vbar14 = fft_vbar + dt / 4 * champubarv
            fft_ug14 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 4 * champmoyu
            fft_vg14 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 4 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t + dt / 4, fft_ubar14, fft_vbar14, fft_ug14, fft_vg14, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar34 = fft_ubar12 + dt / 4 * champubaru
            fft_vbar34 = fft_vbar12 + dt / 4 * champubarv
            fft_ug34 = fft_ug12 + epsilon * reconstr(ichampgu, (t + 3 * dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 4 * champmoyu
            fft_vg34 = fft_vg12 + epsilon * reconstr(ichampgv, (t + 3 * dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 4 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t + 3 * dt / 4, fft_ubar34, fft_vbar34, fft_ug34, fft_vg34,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar1 = fft_ubar12 + dt / 2 * champubaru
            fft_vbar1 = fft_vbar12 + dt / 2 * champubarv
            fft_ug1 = fft_ug12 + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg1 = fft_vg12 + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_3(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar1bis = fft_ubar + dt * champubaru
            fft_vbar1bis = fft_vbar + dt * champubarv
            fft_ug1bis = fft_ug + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                         - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt * champmoyu
            fft_vg1bis = fft_vg + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                         - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt * champmoyv

            t = t + dt
            fft_ubar = (4 * fft_ubar1 - fft_ubar1bis) / 3
            fft_vbar = (4 * fft_vbar1 - fft_vbar1bis) / 3
            fft_ug = (4 * fft_ug1 - fft_ug1bis) / 3
            fft_vg = (4 * fft_vg1 - fft_vg1bis) / 3

            C2u, C2v = C2(t, fft_ubar, fft_vbar, A1, A2, matr, conjmatr,
                          micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)
            uC2eval = reconstr(C2u, t / epsilon, T, Ntaumm)
            vC2eval = reconstr(C2v, t / epsilon, T, Ntaumm)
            fft_u = uC2eval + fft_ug
            fft_v = vC2eval + fft_vg

        elseif schema_micmac == 4

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar16 = fft_ubar + dt / 6 * champubaru
            fft_vbar16 = fft_vbar + dt / 6 * champubarv
            fft_ug16 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 6) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 6 * champmoyu
            fft_vg16 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 6) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 6 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 6, fft_ubar16, fft_vbar16, fft_ug16, fft_vg16,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar13 = fft_ubar + dt / 3 * champubaru
            fft_vbar13 = fft_vbar + dt / 3 * champubarv
            fft_ug13 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 3) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 3 * champmoyu
            fft_vg13 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 3) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 3 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 3, fft_ubar13, fft_vbar13, fft_ug13, fft_vg13,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar12 = fft_ubar13 + dt / 6 * champubaru
            fft_vbar12 = fft_vbar13 + dt / 6 * champubarv
            fft_ug12 = fft_ug13 + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, (t + dt / 3) / epsilon, T, Ntaumm) + dt / 6 * champmoyu
            fft_vg12 = fft_vg13 + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, (t + dt / 3) / epsilon, T, Ntaumm) + dt / 6 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar23 = fft_ubar13 + dt / 3 * champubaru
            fft_vbar23 = fft_vbar13 + dt / 3 * champubarv
            fft_ug23 = fft_ug13 + epsilon * reconstr(ichampgu, (t + 2 * dt / 3) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, (t + dt / 3) / epsilon, T, Ntaumm) + dt / 3 * champmoyu
            fft_vg23 = fft_vg13 + epsilon * reconstr(ichampgv, (t + 2 * dt / 3) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, (t + dt / 3) / epsilon, T, Ntaumm) + dt / 3 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + 2 * dt / 3, fft_ubar23, fft_vbar23, fft_ug23, fft_vg23,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar56 = fft_ubar23 + dt / 6 * champubaru
            fft_vbar56 = fft_vbar23 + dt / 6 * champubarv
            fft_ug56 = fft_ug23 + epsilon * reconstr(ichampgu, (t + 5 * dt / 6) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, (t + 2 * dt / 3) / epsilon, T, Ntaumm) + dt / 6 * champmoyu
            fft_vg56 = fft_vg23 + epsilon * reconstr(ichampgv, (t + 5 * dt / 6) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, (t + 2 * dt / 3) / epsilon, T, Ntaumm) + dt / 6 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + 5 * dt / 6, fft_ubar56, fft_vbar56, fft_ug56, fft_vg56,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar0 = fft_ubar23 + dt / 3 * champubaru
            fft_vbar0 = fft_vbar23 + dt / 3 * champubarv
            fft_ug0 = fft_ug23 + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgu, (t + 2 * dt / 3) / epsilon, T, Ntaumm) + dt / 3 * champmoyu
            fft_vg0 = fft_vg23 + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgv, (t + 2 * dt / 3) / epsilon, T, Ntaumm) + dt / 3 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2, matr, conjmatr,
                         micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar14 = fft_ubar + dt / 4 * champubaru
            fft_vbar14 = fft_vbar + dt / 4 * champubarv
            fft_ug14 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 4 * champmoyu
            fft_vg14 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 4 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 4, fft_ubar14, fft_vbar14, fft_ug14, fft_vg14, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar34 = fft_ubar12 + dt / 4 * champubaru
            fft_vbar34 = fft_vbar12 + dt / 4 * champubarv
            fft_ug34 = fft_ug12 + epsilon * reconstr(ichampgu, (t + 3 * dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 4 * champmoyu
            fft_vg34 = fft_vg12 + epsilon * reconstr(ichampgv, (t + 3 * dt / 4) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 4 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + 3 * dt / 4, fft_ubar34, fft_vbar34, fft_ug34, fft_vg34,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)

            fft_ubar1 = fft_ubar12 + dt / 2 * champubaru
            fft_vbar1 = fft_vbar12 + dt / 2 * champubarv
            fft_ug1 = fft_ug12 + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg1 = fft_vg12 + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t, fft_ubar, fft_vbar, fft_ug, fft_vg, A1, A2,
                         matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar12 = fft_ubar + dt / 2 * champubaru
            fft_vbar12 = fft_vbar + dt / 2 * champubarv
            fft_ug12 = fft_ug + epsilon * reconstr(ichampgu, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt / 2 * champmoyu
            fft_vg12 = fft_vg + epsilon * reconstr(ichampgv, (t + dt / 2) / epsilon, T, Ntaumm) \
                       - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt / 2 * champmoyv

            champubaru, champubarv, ichampgu, ichampgv, champmoyu, champmoyv = \
                champs_4(t + dt / 2, fft_ubar12, fft_vbar12, fft_ug12, fft_vg12,
                         A1, A2, matr, conjmatr, micmac.data.sigma, micmac.data.llambda,
                         Ktaubis, epsilon, Ntaumm)

            fft_ubar2 = fft_ubar + dt * champubaru
            fft_vbar2 = fft_vbar + dt * champubarv
            fft_ug2 = fft_ug + epsilon * reconstr(ichampgu, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgu, t / epsilon, T, Ntaumm) + dt * champmoyu
            fft_vg2 = fft_vg + epsilon * reconstr(ichampgv, (t + dt) / epsilon, T, Ntaumm) \
                      - epsilon * reconstr(ichampgv, t / epsilon, T, Ntaumm) + dt * champmoyv

            t = t + dt
            fft_ubar = (27 * fft_ubar0 - 16 * fft_ubar1 + fft_ubar2) / 12
            fft_vbar = (27 * fft_vbar0 - 16 * fft_vbar1 + fft_vbar2) / 12
            fft_ug = (27 * fft_ug0 - 16 * fft_ug1 + fft_ug2) / 12
            fft_vg = (27 * fft_vg0 - 16 * fft_vg1 + fft_vg2) / 12

            (C3u, C3v) = C3(t, fft_ubar, fft_vbar, A1, A2, matr, conjmatr,
                            micmac.data.sigma, micmac.data.llambda, Ktaubis, epsilon, Ntaumm)
            uC3eval = reconstr(C3u, t / epsilon, T, Ntaumm)
            vC3eval = reconstr(C3v, t / epsilon, T, Ntaumm)
            fft_u = uC3eval + fft_ug
            fft_v = vC3eval + fft_vg

        end

    end 

    fft_u = exp(1j * sqrt(1 + epsilon * k^2) * t / epsilon) * fft_u
    fft_v = exp(1j * sqrt(1 + epsilon * k^2) * t / epsilon) * fft_v

    u = ifft(fft_u)
    v = ifft(fft_v)

    u, v

end

