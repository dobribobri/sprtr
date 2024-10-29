from typing import Union
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from core.planck import planck, BlackBody, Body
from core.session import Session, Stages, Channel
import core.initials as initials


def planck_001(eps: Union[float, np.ndarray, dict] = 1):
    plt.figure()
    wavelengths = np.linspace(135, 3000, 1000) / 1000  # мкм

    body = Body(eps)

    for _T, c in zip([4000, 5000, 6000, 7000], ['r', 'g', 'b', 'black']):
        plt.plot(wavelengths * 1000., body.intensity(wavelengths, _T) / np.pi,
                 c=c, ls='-', label='{} K, модель'.format(_T))

    from radis import planck

    for _T, c in zip([4000, 5000, 6000, 7000], ['r', 'g', 'b', 'black']):
        plt.plot(wavelengths * 1000., planck(wavelengths * 1000., _T),
                 c=c, ls='--', label='{} K, radis'.format(_T))

    plt.xlabel(r'Длина волны $\lambda$, нм')
    plt.ylabel(r'Интенсивность I, Вт/ср/см$^2$/мкм')

    plt.legend(loc='best', frameon=False)
    plt.show()


def planck_002():
    eps = {1: 1, 2: 1., 2.5: 4}
    planck_001(eps)


def planck_003():
    body = Body(1)
    lambdas = np.array([0.45, 0.6, 0.8, 1.0, 1.2, 1.38, 1.61, 1.8, 2.0, 2.15])[::-1]
    T_range = [1000, 1500, 1773, 2000, 2500, 4000]
    data = []
    for n, wavelength in enumerate(lambdas):
        intensities = [body.intensity(wavelength, T=T) for T in T_range]
        data.append([n + 1,
                     wavelength,
                     *intensities])
    data = np.asarray(data)
    data = pd.DataFrame(data=data[:, 1:], index=data[:, 0],
                        columns=["мкм"] + ['{} K'.format(T) for T in T_range])
    data.to_excel('planck_003.xlsx')


def planck_004():
    plt.figure()
    wl = np.linspace(0.1, 3, 100)
    plt.plot(wl, planck(wl, 3000))

    Ldata = np.asarray([0.135, 1.1, 2])
    Idata = planck(Ldata, np.asarray([3000, 2990, 3010]))

    plt.scatter(Ldata, Idata)

    body = BlackBody()
    T = body.temperature(Ldata, Idata)
    plt.title('T = {}'.format(T))
    plt.plot(wl, planck(wl, T))
    plt.show()


def planck_005(eps: Union[float, np.ndarray, dict] = 1):
    body = Body(eps=eps)

    plt.figure()
    wl = np.linspace(0.1, 3, 100)
    plt.plot(wl, body.intensity(wl, 3000))

    Ldata = np.asarray([0.135, 1.1, 2])
    Idata = body.intensity(Ldata, np.asarray([3000, 2990, 3010]))

    plt.scatter(Ldata, Idata)

    T = body.temperature(Ldata, Idata)
    plt.title('T = {}'.format(T))
    plt.plot(wl, body.intensity(wl, T))
    plt.show()


def planck_006():
    planck_005(eps=0.5)


def planck_007():
    planck_005(eps={1: 1, 2: 1., 2.5: 4})


def planck_008():
    body = Body(1)
    lambdas = np.array([0.45, 0.6, 0.8, 1.0, 1.16, 1.2, 1.38, 1.61, 1.8, 2.0, 2.15])
    T = 2500
    _I = [body.intensity(wavelength=wavelength, T=T) for wavelength in lambdas]
    i = np.argmax(_I)
    print(lambdas[i])
    print(np.asarray(list(zip(lambdas, _I, _I / np.max(_I)))[::-1]))


def session_001():
    wavelengths = np.asarray(initials.configuration)[:, 1]
    session = Session(channels=list([Channel(wavelength=wavelength) for wavelength in wavelengths]))

    directory = './2023_11_28/iskra'
    filenames = ['C{}--opyt.txt'.format(i) for i in range(1, 5)]
    paths = [os.path.join(directory, filename) for filename in filenames]

    session.read_channels(filepaths=paths, format_='txt')

    session.mask_indexes = [1, 2]
    session.set_timedelta()
    plt.figure()
    for i, channel in enumerate(session.channels_masked):
        channel.board = 1
        channel.number = i + 1
        plt.plot(channel.time_synced, channel.data,
                 label='Устройство #{}, канал #{}'.format(channel.board, channel.number))
    plt.xlabel('Время')
    plt.ylabel('АЦП (?)')
    plt.legend(loc='best', frameon=True)
    plt.show()

    time, T = session.get_temperature(t_start=1e-5, t_stop=2e-5, parallel=True)

    plt.figure()
    for i, channel in enumerate(session.channels_masked):
        channel.board = 1
        channel.number = i + 1
        plt.plot(channel.time_synced, channel.data,
                 label='Устройство #{}, канал #{}'.format(channel.board, channel.number))
    plt.xlabel('Время')
    plt.ylabel(r'АЦП * коэфф. усиления * $\alpha$ (?)')
    plt.legend(loc="upper right", frameon=False)

    ax = plt.gca()
    ax = plt.twinx(ax)
    ax.plot(time, T, color='blue', lw=2, label='Температура')

    plt.legend(loc="center right", frameon=False)
    plt.show()
