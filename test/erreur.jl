function erreur(u, v, epsilon, dataset):
    # ici la reference a été calculee par micmac

    str0 = ['', 'donnees_cubique_128_micmac/',
            'donnees_FS_128_micmac/',
            'donnees_data3_128_micmac/']

    str3 = 'donnee_'
    str5 = '.txt'

    str4 = {
    10:       '10',
    5:        '5',
    2.5:      '2_5',
    1:        '1',
    0.5:      '0_5',
    0.2:      '0_2',
    0.25:     '0_25',
    0.1:      '0_1',
    0.05:     '0_05',
    0.025:    '0_025',
    0.01:     '0_01',
    0.005:    '0_005',
    0.0025:   '0_0025',
    0.001:    '0_001',
    0.0005:   '0_0005',
    0.00025:  '0_00025',
    0.0001:   '0_0001',
    0.00005:  '0_00005',
    0.000025: '0_000025',
    0.00001:  '0_00001',
    0.000005: '0_000005',
    0.0000025: '0_0000025',
    0.000001: '0_000001'

    }
    fichier = os.path.join(str0[dataset], str3 + str4[epsilon] + str5)

    uv = sp.zeros((4, 128))

    open(fichier, "r") do fichier

        for j in range(128):
            uv[0, j] = float(azerty.read(25))
            uv[1, j] = float(azerty.read(25))
            uv[2, j] = float(azerty.read(25))
            uv[3, j] = float(azerty.read(25))
            azerty.read(1)
        end

    end

    if dataset == 1
        NN = sp.shape(u)[1]
        xmin = -8
        xmax = 8
        t = 0.4
    elseif dataset == 2
        NN = sp.shape(u)[1]
        xmin = 0
        xmax = 2 * pi
        t = 1
    elseif dataset == 3
        NN = sp.shape(u)[1]
        xmin = 0
        xmax = 2 * pi
        T = 2 * pi
        t = 0.25
    end

    dx = (xmax - xmin) / NN
    x = sp.linspace(xmin, xmax - dx, NN)[None]
    dx = x[0, 1] - x[0, 0]
    k = 2 * pi / (xmax - xmin) * np.concatenate((np.arange(0, NN / 2, 1),
                                                 np.arange(NN / 2 - NN, 0, 1)), dims=1)

    if sp.shape(uv)[1] != NN
        uv = reconstr_x(uv, x, xmin, xmax)
    end

    uvu = uv[0, :] + 1im * uv[1, :]
    uvv = uv[2, :] + 1im * uv[3, :]

    refH1 = sp.sqrt(dx * np.linalg.norm(ifft(1im * sp.sqrt(1 + k .^ 2) * fft(uvu))) .^ 2 + dx * np.linalg.norm(
        ifft(1im * sp.sqrt(1 + k .^ 2) * fft(uvv))) .^ 2)
    err = (sp.sqrt(dx * np.linalg.norm(ifft(1im * sp.sqrt(1 + k .^ 2) * fft(u - uvu))) .^ 2 + dx * np.linalg.norm(
        ifft(1im * sp.sqrt(1 + k .^ 2) * fft(v - uvv))) .^ 2)) / refH1

    return err

end
