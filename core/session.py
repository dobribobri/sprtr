from typing import Union, List, Tuple
import numpy as np
import dill
import tqdm
from multiprocessing import Pool
import core.initials as initials
from core.planck import Body


class Channel:
    def __init__(self, wavelength: float, timeseries: np.ndarray = None, board: int = None, number: int = None):
        self.board = board  # номер платы
        self.number = number  # номер канала
        self.wavelength = wavelength  # длина волны
        if timeseries is None:
            self.time, self.data = None, None
        else:
            self.time = timeseries[:, 0]  # отсчеты времени
            self.data = timeseries[:, 1]  # значения с АЦП
        self.gain = 1.  # коэффициент усиления
        self.alpha = 1.  # калибровочный коэффициент, переводящий уровни с АЦП в интенсивности АЧТ
        self.timedelta = 0.  # смещение по времени

    @property
    def data_gained(self):
        return self.gain * self.data

    @property
    def data_calibrated(self):
        return self.alpha * self.gain * self.data

    @property
    def time_synced(self):
        return self.time - self.timedelta

    def find_peak(self):
        peak = np.argmax(self.data)
        return self.time[peak], self.data[peak]

    def erase(self):
        self.time = None
        self.data = None

    def read(self, filepath: str, format_='txt', n_interp: Union[int, None] = 4096):
        if format_ == 'txt':
            self.read_txt(filepath, n_interp)

    def read_txt(self, filepath: str, n_interp: Union[int, None] = 4096):
        data = np.loadtxt(filepath)
        if data.shape != (len(data), 2):
            raise "Неверный формат данных"

        if n_interp is None:
            self.time, self.data = data[:, 0], data[:, 1]
            return

        t_start = data[0, 0]
        t_stop = data[-1, 0]
        self.time = np.linspace(t_start, t_stop, n_interp)
        self.data = np.interp(self.time, data[:, 0], data[:, 1])


class Session:
    def __init__(self, channels: List[Channel] = None):
        self.channels: List[Channel] = channels

        # Соответствие номера канала и его длины волны
        self.configuration = initials.configuration

        # Число точек интерполяции
        self.n_interp_rf: Union[int, None] = 4096  # при чтении данных из файла
        self.n_interp_tp: Union[int, None] = 1000  # при расчете значений температуры

        # Настройка коэффициента усиления
        self.T_gain_cal = 2500                        # температура эталона

        # Абсолютная калибровка
        self.T_abs_cal = 2500                         # температура эталона
        self.eps_abs_cal = 1                          # излучательная способность эталона

        # Относительная калибровка
        self.T_rel_cal = 2500                         # температура эталона
        self.eps_rel_cal = 1                          # излучательная способность эталона

        # Измерение образца
        self.eps_sample = 1.                          # излучательная способность образца

        # Кол-во ядер
        self.n_workers = 8

        # Данные, направленные на обработку
        self.__data = None

    @property
    def n_channels(self) -> int:
        """
        Количество каналов
        """
        return len(self.channels)

    @property
    def wavelengths(self) -> np.ndarray:
        """
        Используемые длины волн
        """
        return np.asarray([channel.wavelength for channel in self.channels])

    @property
    def sample(self) -> Body:
        """
        Исследуемый образец
        """
        return Body(eps=self.eps_sample)

    def set_configuration(self, configuration: List[Tuple[int, float]]):
        """
        Установить соответствия номер канала <--> длина волны
        """
        self.configuration = configuration

    def read_channels(self, filepaths: list, format_='txt'):
        """
        Прочитать данные из файлов filepaths и записать в каналы в соответствии с конфигурацией
        """
        self.channels = []
        for i, filepath in enumerate(filepaths):
            number, wavelength = self.configuration[i]
            self.read_channel(wavelength, filepath=filepath, format_=format_, number=number)

    def read_channel(self, wavelength: float, filepath: str, format_='txt', board: int = None, number: int = None):
        """
        Прочитать данные из файла filepath и записать в новый канал
        """
        channel = Channel(wavelength=wavelength, board=board, number=number)
        channel.read(filepath, format_=format_, n_interp=self.n_interp_rf)
        self.channels.append(channel)

    @staticmethod
    def __goal_level(wavelength: float, T: float):
        return initials.bb_adc_levels[np.round(wavelength, decimals=2)][np.round(T, decimals=0)]

    def set_gain(self, values: Union[list, np.ndarray] = None):
        """
        Установка коэффициента усиления.
        Если values = None, коэффициенты усиления устанавливаются в соответствии с эталонными уровнями в initials
        """
        for i in range(self.n_channels):
            if values is None:
                goal_level = Session.__goal_level(wavelength=self.channels[i].wavelength, T=self.T_gain_cal)
                self.channels[i].gain = goal_level / np.mean(self.channels[i].data)
            else:
                self.channels[i].gain = values[i]

    def absolute_calibration(self):
        """
        Абсолютная калибровка
        """
        body = Body(eps=self.eps_abs_cal)
        for i in range(self.n_channels):
            goal_intensity = body.intensity(wavelength=self.channels[i].wavelength, T=self.T_abs_cal)
            self.channels[i].alpha = goal_intensity / np.mean(self.channels[i].data_gained)

    def relative_calibration(self):
        """
        Относительная калибровка
        """
        body = Body(eps=self.eps_rel_cal)
        I_spectrum = [body.intensity(wavelength=wavelength, T=self.T_rel_cal) for wavelength in self.wavelengths]
        i_max = np.argmax(I_spectrum)
        for i in range(len(self.channels)):
            self.channels[i].alpha = I_spectrum[i] / I_spectrum[i_max] * \
                                     np.mean(self.channels[i_max].data_gained) / np.mean(self.channels[i].data_gained)

    def set_timedelta(self):
        """
        Вычислить смещения каналов по времени
        """
        t0, _ = self.channels[0].find_peak()
        self.channels[0].timedelta = 0.
        for i in range(1, len(self.channels)):
            t1, _ = self.channels[i].find_peak()
            self.channels[i].timedelta = t1 - t0

    def get_boundaries(self):
        """
        Временной интервал измерений спектров
        """
        self.set_timedelta()
        bounds = np.asarray([(channel.time_synced[0], channel.time_synced[-1]) for channel in self.channels])
        left, right = np.max(bounds[:, 0]), np.min(bounds[:, 1])
        return left, right

    def process(self, i):
        return i, self.sample.temperature(wavelengths=self.wavelengths, intensities=self.__data[i])

    def get_temperature(self, t_start: float = None, t_stop: float = None,
                        n_interp: int = None,
                        parallel: bool = True):
        """
        Расчет температуры по Планку
        :return:
        """
        left, right = self.get_boundaries()
        if t_start is None:
            t_start = left
        if t_stop is None:
            t_stop = right
        if n_interp is None:
            n_interp = self.n_interp_tp

        data = []
        lengths = []
        for channel in self.channels:
            cond = (t_start <= channel.time_synced) & (channel.time_synced <= t_stop)
            lengths.append(np.count_nonzero(cond))
            data.append([channel.time_synced[cond], channel.data_calibrated[cond]])

        if n_interp is None:
            n_interp = np.min(lengths)

        time = np.linspace(t_start, t_stop, n_interp)
        for i, channel in enumerate(self.channels):
            data[i] = np.interp(time, data[i][0], data[i][1])
        self.__data = np.asarray(data).T

        if not parallel:
            T = []
            for spectrum in self.__data:
                T.append(self.sample.temperature(wavelengths=self.wavelengths, intensities=spectrum))
            return time, np.asarray(T)

        results = []
        n = len(self.__data)
        with Pool(processes=self.n_workers) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(self.process, range(n)), total=n):
                results.append(result)

        results = np.asarray(sorted(results, key=lambda e: e[0]))
        return time, results[:, 1]  # T

    def save(self, filename='session'):
        """
        Сохранить сессию
        """
        with open(filename, 'wb') as dump:
            dill.dump(self, dump)

    @classmethod
    def load(cls, filename='session'):
        """
        Загрузить сессию
        """
        with open(filename, 'rb') as dump:
            return dill.load(dump)

    def clear(self):
        """
        Очистить сессию
        """
        self.__init__()
        self.save()

    def erase(self):
        """
        Очистить данные каналов
        """
        for channel in self.channels:
            channel.erase()


if __name__ == "__main__":
    import core.test
    core.test.session_001()
