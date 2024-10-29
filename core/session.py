from typing import Union, List
from enum import Enum
import numpy as np
import json
import tqdm
from multiprocessing import Pool
import core.initials as initials
from core.planck import Body
import core.tekwfm as tekwfm
# import xml.etree.ElementTree as et


class Stages(Enum):
    TimeSynchro = 0  # синхронизация каналов по времени
    Calibration = 1  # калибровка каналов
    Measurement = 2  # проведение эксперимента


class Series:
    def __init__(self):
        self.time: np.ndarray = None   # отсчеты времени
        self.data: np.ndarray = None   # значения

        self.mask: bool = False         # применять обработку
        self.n_interp: Union[int, None] = 1024      # число точек интерполяции при загрузке из файла

    @staticmethod
    def interpolate(time: np.asarray, data: np.asarray, n_interp: int = None):
        if n_interp is None:
            return time, data
        new_time = np.linspace(time[0], time[-1], n_interp)
        data = np.interp(new_time, time, data)
        return new_time, data


# noinspection PyTypeChecker
class Channel:
    def __init__(self, wavelength: float,
                 board: int = None, number: int = None,
                 used: bool = True, current_stage: Stages = Stages.Measurement):
        self.used = used  # использовать канал?
        self.current_stage = current_stage  # текущий этап

        self.board = board  # номер платы
        self.number = number  # номер канала
        self.wavelength = wavelength  # длина волны

        # 0 - TimeSynchro Series, 1 - Calibration Series, 2 - Measurement Series
        self.Series = [Series()] * 3
        # 0 - данные сеанса для определения временных сдвигов
        # 1 - данные калибровочного сеанса
        # 2 - данные эксперимента

        self.gain = 1.  # коэффициент усиления
        self.alpha = 1.  # калибровочный коэффициент, переводящий уровни с АЦП в интенсивности АЧТ
        self.timedelta = 0.  # смещение по времени

    @property
    def series(self) -> Series:
        return self.Series[self.current_stage.value]

    @property
    def data(self) -> np.ndarray:
        return self.series.data

    @data.setter
    def data(self, _data: np.ndarray):
        self.Series[self.current_stage.value].data = _data

    @property
    def time(self) -> np.ndarray:
        return self.series.time

    @time.setter
    def time(self, _time: np.ndarray):
        self.Series[self.current_stage.value].time = _time

    @property
    def data_gained(self) -> np.ndarray:
        return self.gain * self.data

    @property
    def data_calibrated(self) -> np.ndarray:  # калиброванные данные
        return self.alpha * self.gain * self.data

    @property
    def time_synced(self) -> np.ndarray:  # ход времени с учетом смещения
        return self.time - self.timedelta

    def find_peak(self) -> tuple:
        peak = np.argmax(self.data)
        return self.time[peak], self.data[peak]

    @property
    def mask(self) -> bool:
        return self.series.mask

    @mask.setter
    def mask(self, _mask: bool):
        self.Series[self.current_stage.value].mask = _mask

    @property
    def n_interp(self) -> Union[int, None]:
        return self.series.n_interp

    @n_interp.setter
    def n_interp(self, _val: Union[int, None]):
        self.Series[self.current_stage.value].n_interp = _val

    def erase_current(self) -> None:
        self.Series[self.current_stage.value] = Series()

    def clear_all(self) -> None:
        self.Series = [Series()] * 3

    def read(self, filepath: str, format_='txt'):
        match format_:
            case 'txt':
                self.read_txt(filepath)
            case 'wfm':
                self.read_wfm(filepath)
            case _:
                self.read_txt(filepath)

    def read_txt(self, filepath: str):
        series = np.asarray(np.loadtxt(filepath))
        if series.shape != (len(series), 2):
            raise "Неверный формат данных"

        self.time, self.data = \
            Series.interpolate(time=series[:, 0], data=series[:, 1], n_interp=self.n_interp)

    def read_wfm(self, filepath: str, frameNo=0):
        try:
            volts, tstart, tscale, tfrac, tdatefrac, tdate = tekwfm.read_wfm(filepath)
        except tekwfm.WfmReadError:
            raise "Ошибка чтения .wfm"

        # create time vector
        samples, frames = volts.shape
        tstop = samples * tscale + tstart
        t = np.linspace(tstart, tstop, num=samples, endpoint=False)

        # fractional trigger
        times = np.zeros(volts.shape)
        for frame, subsample in enumerate(tfrac):
            toff = subsample * tscale
            times[:, frame] = t + toff
        if frameNo >= frames:
            frameNo = 0
        gt = times[:, frameNo]
        if frameNo != -1:
            gw = volts[:, frameNo]
        else:
            gw = np.average(volts, axis=1)

        self.time, self.data = \
            Series.interpolate(time=gt, data=gw, n_interp=self.n_interp)


class Session:
    def __init__(self, channels: List[Channel] = None):
        self.channels: List[Channel] = channels  # Каналы

        # Настройка коэффициента усиления
        self.T_gain_cal = 2500                        # температура эталона

        # Калибровка
        # Абсолютная калибровка
        self.T_abs_cal = 2500                         # температура эталона
        self.eps_abs_cal = 1                          # излучательная способность эталона
        # Относительная калибровка
        self.T_rel_cal = 2500                         # температура эталона
        self.eps_rel_cal = 1                          # излучательная способность эталона

        # Измерение образца
        self.eps_sample = 1.                          # излучательная способность образца
        self.tp_n_interp_exp: Union[int, None] = 100               # число точек интерполяции при расчете

        # Кол-во ядер
        self.n_workers = 8

        # Данные, направленные на обработку
        self.__data = None

    @property
    def first_channel(self):
        return self.channels[0]

    @property
    def last_channel(self):
        return self.channels[-1]

    @property
    def stage(self):
        s = self.channels[0].current_stage
        for channel in self.channels:
            if s.value != channel.current_stage.value:
                return None
        return s

    @stage.setter
    def stage(self, _stage: Stages = Stages.Measurement):
        for i in range(len(self.channels)):
            self.channels[i].current_stage = _stage

    @property
    def used_indexes(self) -> list:
        """
        Индексы используемых каналов
        """
        if self.channels:
            return [i for i, channel in enumerate(self.channels) if channel.used]
        return []

    @used_indexes.setter
    def used_indexes(self, indexes):
        for i in range(len(self.channels)):
            if i in indexes:
                self.channels[i].used = True
            else:
                self.channels[i].used = False
                self.channels[i].Series[Stages.TimeSynchro.value].mask = False
                self.channels[i].Series[Stages.Calibration.value].mask = False
                self.channels[i].Series[Stages.Measurement.value].mask = False

    @property
    def used(self) -> np.ndarray:
        arr = []
        for i in range(len(self.channels)):
            arr.append(i in self.used_indexes)
        return np.asarray(arr)

    @used.setter
    def used(self, _mask):
        for i in range(len(self.channels)):
            self.channels[i].used = _mask[i]

    @property
    def mask_indexes(self) -> list:
        mask_ = []
        for i in self.used_indexes:
            if self.channels[i].mask:
                mask_.append(i)
        return mask_

    @mask_indexes.setter
    def mask_indexes(self, indexes):
        for i in self.used_indexes:
            self.channels[i].mask = i in indexes

    @property
    def mask(self) -> np.ndarray:
        arr = []
        for i in range(len(self.channels)):
            arr.append(i in self.mask_indexes)
        return np.asarray(arr)

    @mask.setter
    def mask(self, _mask):
        for i in range(len(self.channels)):
            self.channels[i].used = (_mask[i] and (i in self.used_indexes))

    @property
    def wavelengths_masked(self) -> np.ndarray:
        return np.asarray([self.channels[i].wavelength for i in self.mask_indexes])

    @property
    def channels_masked(self) -> list:
        return [self.channels[i] for i in self.mask_indexes]

    @property
    def sample(self) -> Body:
        """
        Исследуемый образец
        """
        return Body(eps=self.eps_sample)

    def read_channels(self,
                      filepaths: Union[list, str],
                      format_='txt',
                      ):
        """
        Прочитать данные из файлов filepaths и записать в каналы в соответствии с конфигурацией
        """

        if format_ in ['txt', 'wfm']:

            for i, filepath in enumerate(filepaths):
                self.read_channel(index=i, filepath=filepath, format_=format_)

        elif format_ in ['dat', 'csv']:
            k = 0
            for j, filepath in enumerate(filepaths):
                series = []
                match format_:
                    case 'dat':
                        series = np.loadtxt(filepath)
                    case 'csv':
                        series = np.loadtxt(filepath, delimiter=',', skiprows=1)
                series = np.asarray(series)
                n_channels = series.shape[1] - 1
                for i in range(n_channels):

                    self.channels[k + i].time, self.channels[k + i].data = \
                        Series.interpolate(series[:, 0], series[:, i + 1], self.channels[k + i].n_interp)
                    self.channels[k + i].mask = True
                k += n_channels

        else:  # TBD .xml
            pass

    def read_channel(self,
                     index: int,
                     filepath: str,
                     format_: str = 'txt',
                     ):
        """
        Прочитать данные из файла filepath и записать в новый канал
        """

        self.channels[index].read(filepath=filepath, format_=format_)
        self.channels[index].mask = True

    @staticmethod
    def __goal_level(wavelength: float, T: float):
        """
        Эталонный уровень сигнала с АЦП при измерении АЧТ с заданной температурой на данной длине волны
        """
        return initials.bb_adc_levels[np.round(wavelength, decimals=2)][np.round(T, decimals=0)]

    def set_gain(self, values: Union[list, np.ndarray] = None):
        """
        Установка коэффициента усиления.
        Если values = None, коэффициенты усиления устанавливаются в соответствии с эталонными уровнями в initials
        """
        for i, index in enumerate(self.used_indexes):
            if values is None:
                goal_level = Session.__goal_level(wavelength=self.channels[index].wavelength, T=self.T_gain_cal)
                self.channels[index].gain = goal_level / np.mean(self.channels[index].data)
            else:
                self.channels[index].gain = values[i]

    def absolute_calibration(self):
        """
        Абсолютная калибровка
        """
        body = Body(eps=self.eps_abs_cal)
        for index in self.mask_indexes:
            goal_intensity = body.intensity(wavelength=self.channels[index].wavelength, T=self.T_abs_cal)
            self.channels[index].alpha = goal_intensity / np.mean(self.channels[index].data_gained)

    # noinspection PyTypeChecker
    def relative_calibration(self):
        """
        Относительная калибровка
        """
        body = Body(eps=self.eps_rel_cal)
        I_spectrum = [body.intensity(wavelength=wavelength, T=self.T_rel_cal) for wavelength in self.wavelengths_masked]
        i_max = np.argmax(I_spectrum)
        index_max = self.mask_indexes[i_max]
        for i, index in enumerate(self.mask_indexes):
            self.channels[index].alpha = I_spectrum[i] / I_spectrum[i_max] * \
                np.mean(self.channels[index_max].data_gained) / np.mean(self.channels[index].data_gained)

    def set_timedelta(self):
        """
        Вычислить смещения каналов по времени
        """
        first = self.mask_indexes[0]
        t0, _ = self.channels[first].find_peak()
        self.channels[first].timedelta = 0.
        for i in self.mask_indexes[1:]:
            t1, _ = self.channels[i].find_peak()
            self.channels[i].timedelta = t1 - t0

    def process(self, i):
        return i, self.sample.temperature(wavelengths=self.wavelengths_masked, intensities=self.__data[i])

    def get_temperature(self, t_start: float = None, t_stop: float = None, parallel: bool = True):
        """
        Расчет температуры по Планку
        """
        bounds = np.asarray([(self.channels[i].time_synced[0], self.channels[i].time_synced[-1])
                             for i in self.mask_indexes])
        left, right = np.max(bounds[:, 0]), np.min(bounds[:, 1])

        if t_start is None:
            t_start = left
        if t_stop is None:
            t_stop = right

        data = []
        lengths = []
        for i in self.mask_indexes:
            cond = (t_start <= self.channels[i].time_synced) & (self.channels[i].time_synced <= t_stop)
            lengths.append(np.count_nonzero(cond))
            data.append([self.channels[i].time_synced[cond], self.channels[i].data_calibrated[cond]])

        n_interp = self.tp_n_interp_exp
        if n_interp is None:
            n_interp = np.min(lengths)

        time = np.linspace(t_start, t_stop, n_interp)
        for i, _ in enumerate(self.mask_indexes):
            data[i] = np.interp(time, data[i][0], data[i][1])
        self.__data = np.asarray(data).T

        if not parallel:
            T = []
            for spectrum in self.__data:
                T.append(self.sample.temperature(wavelengths=self.wavelengths_masked, intensities=spectrum))
            return time, np.asarray(T)

        results = []
        n = len(self.__data)
        with Pool(processes=self.n_workers) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(self.process, range(n)), total=n):
                results.append(result)

        results = np.asarray(sorted(results, key=lambda e: e[0]))
        return time, results[:, 1]  # T

    def save(self, filepath='session') -> None:
        """
        Сохранить сессию
        """
        info = {'channels': []}

        for attr_name in ['T_gain_cal', 'T_abs_cal', 'eps_abs_cal', 'T_rel_cal', 'eps_rel_cal',
                          'eps_sample', 'tp_n_interp_exp', 'n_workers']:
            info[attr_name] = getattr(self, attr_name)

        for channel in self.channels:
            fields = {'series': [],
                      'current_stage': channel.current_stage.value}

            for attr_name in ['used', 'board', 'number', 'wavelength', 'gain', 'alpha', 'timedelta']:
                fields[attr_name] = getattr(channel, attr_name)

            for series in channel.Series:
                fields_series = {}
                for attr_name in ['time', 'data', 'mask', 'n_interp']:
                    fields_series[attr_name] = getattr(series, attr_name)

                fields['series'].append(fields_series)

            info['channels'].append(fields)

        with open(filepath, 'w') as dump:
            json.dump(info, dump)

    @classmethod
    def load(cls, filepath='session') -> 'Session':
        """
        Загрузить сессию
        """
        with open(filepath, 'r') as dump:
            info = json.load(dump)

        session = Session()

        for attr_name in ['T_gain_cal', 'T_abs_cal', 'eps_abs_cal', 'T_rel_cal', 'eps_rel_cal',
                          'eps_sample', 'tp_n_interp_exp', 'n_workers']:
            setattr(session, attr_name, info[attr_name])

        n_channels = len(info['channels'])
        Channels = [Channel(wavelength=info['channels'][i]['wavelength']) for i in range(n_channels)]
        for i in range(n_channels):
            Channels[i].current_stage = Stages(info['channels'][i]['current_stage'])

            for attr_name in ['used', 'board', 'number', 'wavelength', 'gain', 'alpha', 'timedelta']:
                setattr(Channels[i], attr_name, info['channels'][i][attr_name])

            n_series = len(info['channels'][i]['series'])
            _Series = []
            for j in range(n_series):
                series = Series()
                for attr_name in ['time', 'data', 'mask', 'n_interp']:
                    setattr(series, attr_name, info['channels'][i]['series'][j][attr_name])
                _Series.append(series)

            Channels[i].Series = _Series

        session.channels = Channels
        return session

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
            channel.erase_current()


if __name__ == "__main__":
    import core.test
    core.test.session_001()
