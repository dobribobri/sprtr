from typing import Union, List
import numpy as np
import dill
import core.initials as initials
from core.planck import Body


class Channel:
    def __init__(self, wavelength: float, timeseries: np.ndarray = None, board: int = None, number: int = None):
        self.board = board
        self.number = number
        self.wavelength = wavelength
        if timeseries is None:
            self.time, self.data = None, None
        else:
            self.time = timeseries[:, 0]
            self.data = timeseries[:, 1]
        self.gain = 1.
        self.alpha = 1.
        self.timedelta = 0.
        self.t_start = 0.
        self.t_stop = np.inf

    @property
    def data_calibrated(self):
        return self.alpha * self.gain * self.data

    @property
    def time_synced(self):
        return self.time - self.timedelta

    def find_peak(self):
        peak = np.argmax(self.data)
        return self.time[peak], self.data[peak]

    def crop(self):
        cond = (self.t_start <= self.time_synced) & (self.time_synced <= self.t_stop)
        return self.time_synced[cond], self.data[cond]

    def erase(self):
        self.time = None
        self.data = None

    def read(self, filename: str, format_='txt', n_interp: Union[int, None] = 4096):
        if format_ == 'txt':
            self.read_txt(filename, n_interp)

    def read_txt(self, filename: str, n_interp: Union[int, None] = 4096):
        data = np.loadtxt(filename)
        if data.shape != (len(data), 2):
            raise "Неверный формат данных"

        if n_interp is None:
            self.time, self.data = data[:, 0], data[:, 1]
            return

        t_start = data[0, 0]
        t_end = data[-1, 0]
        self.time = np.linspace(t_start, t_end, n_interp)
        self.data = np.interp(self.time, data[:, 0], data[:, 1])


class Session:
    def __init__(self, channels: List[Channel] = None):
        self.channels = channels

        # Соответствие номера платы, номера канала и его частоты
        self.configuration = initials.configuration

        # Число точек интерполяции при чтении данных из файла
        self.n_interp = 4096

        # Настройка коэффициента усиления
        self.bb_adc_levels = initials.bb_adc_levels   # эталонные уровни сигналов АЦП
        self.T_gain_cal = 2500                        # температура эталона

        # Относительная калибровка
        self.T_rel_cal = 2500                         # температура эталона
        self.eps_rel_cal = 1                          # излучательная способность эталона

        # Измерение образца
        self.eps_sample = 1.                          # излучательная способность образца

    def read_channel(self, wavelength: float, filename: str, format_='txt', board: int = None, number: int = None):
        channel = Channel(wavelength=wavelength, board=board, number=number)
        channel.read(filename, format_=format_, n_interp=self.n_interp)
        self.channels.append(channel)

    def read_channels(self, filenames: str, format_='txt'):
        self.channels = []
        for i, board, number, wavelength in enumerate(self.configuration):
            self.read_channel(wavelength, filename=filenames[i], format_=format_, board=board, number=number)

    def __goal_level(self, wavelength: float, T: float):
        return self.bb_adc_levels[np.round(wavelength, decimals=2)][np.round(T, decimals=0)]

    def set_gain(self):
        for channel in self.channels:
            goal_level = self.__goal_level(wavelength=channel.wavelength, T=self.T_gain_cal)
            channel.gain = goal_level / np.mean(channel.data)

    def set_alpha(self):
        body = Body(eps=self.eps_rel_cal)
        for channel in self.channels:
            goal_intensity = body.intensity(wavelength=channel.wavelength, T=self.T_rel_cal)
            channel.alpha = goal_intensity / np.mean(channel.gain * channel.data)

    def time_sync(self):
        t0, _ = self.channels[0].find_peak()
        self.channels[0].timedelta = 0.
        for i in range(1, len(self.channels)):
            t1, _ = self.channels[i].find_peak()
            self.channels[i].timedelta = t1 - t0
        bounds = np.asarray([(channel.time_synced[0], channel.time_synced[-1]) for channel in self.channels])
        left, right = np.max(bounds[:, 0]), np.min(bounds[:, 1])
        for i in range(len(self.channels)):
            self.channels[i].t_start = left
            self.channels[i].t_stop = right

    def get_temperature(self, t_start: float = None, t_stop: float = None):
        t_begin, t_end = self.get_time_bounds()
        if t_start is None:
            t_start = t_begin
        if t_stop is None:
            t_stop = t_end

        sample = Body(eps=self.eps_sample)
        wavelengths = [channel.wavelength for channel in self.channels]
        for channel in self.channels:
            time, data = channel.get_timeseries()
            cond = (t_start <= time) & (time <= t_stop)
            data = data[cond]


    def save(self, filename='session'):
        with open(filename, 'wb') as dump:
            dill.dump(self, dump)

    @classmethod
    def load(cls, filename='session'):
        with open(filename, 'rb') as dump:
            return dill.load(dump)

    def restore(self):
        self.__init__()
        self.save()

    def erase(self):
        for channel in self.channels:
            channel.erase()


if __name__ == "__main__":
    pass
