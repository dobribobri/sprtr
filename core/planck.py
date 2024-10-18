from typing import Union
import numpy as np
from scipy.optimize import curve_fit

C1 = 37418  # Вт×мкм^4/см^2
C2 = 14388  # мкм×К


def planck(wavelength, T):
    return C1 * np.power(wavelength, -5) / (np.exp(C2 / (wavelength * T)) - 1)


class BlackBody:
    @staticmethod
    def intensity(wavelength: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Спектральная интенсивность излучения абсолютно черного тела
        :param wavelength: длина волны (в мкм)
        :param T: температура (в К)
        :return: Интенсивность (Вт/см^2/мкм)
        """
        return planck(wavelength, T)

    @staticmethod
    def temperature(wavelengths: np.ndarray, intensities: np.ndarray):
        popt, _ = curve_fit(BlackBody.intensity, wavelengths, intensities, bounds=(1e1, 1e4))
        return popt[0]


class Body(BlackBody):
    def __init__(self, eps: Union[float, np.ndarray, dict] = 1.):
        """
        :param eps: коэффициент излучения (излучательная способность)
        """
        self.eps = eps

    def __eps(self, wavelength: Union[float, np.ndarray]):
        if isinstance(self.eps, dict):
            keys = np.asarray(list(self.eps.keys()))

            if isinstance(wavelength, float):
                index = np.argmin(np.abs(keys - wavelength))
                key = keys[index]
                return self.eps[key]

            eps = []
            for _l in wavelength:
                index = np.argmin(np.abs(keys - _l))
                key = keys[index]
                eps.append(self.eps[key])
            return np.asarray(eps)

        return self.eps

    def intensity(self, wavelength: Union[float, np.ndarray], T: Union[float, np.ndarray]):
        """
        Спектральная интенсивность излучения
        :param wavelength: длина волны (в мкм)
        :param T: температура (в К)
        :return: Интенсивность (Вт/см^2/мкм)
        """
        return self.__eps(wavelength) * super().intensity(wavelength, T)

    def temperature(self, wavelengths: np.ndarray, intensities: np.ndarray):
        """
        Определение температуры по интенсивностям
        :param wavelengths: длины волн (в мкм)
        :param intensities: интенсивности (в Вт/см^2/мкм)
        :return: Температура (в К)
        """
        popt, _ = curve_fit(self.intensity, wavelengths, intensities, bounds=(1e1, 1e4))
        return popt[0]


if __name__ == '__main__':
    import core.test

    # core.test.planck_001()
    # core.test.planck_002()
    # core.test.planck_003()
    # core.test.planck_004()
    # core.test.planck_005()
    # core.test.planck_006()
    # core.test.planck_007()
    core.test.planck_008()
