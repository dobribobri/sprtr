from typing import Union, List, Tuple
from enum import Enum
import numpy as np
import json
import tqdm
from multiprocessing import Pool

import core.initials as initials
from core.planck import Body
import core.tekwfm as tekwfm
from core.logger import Log

from scipy.signal import savgol_filter
import scipy.fftpack as fft
from scipy.ndimage import uniform_filter1d, gaussian_filter
import statsmodels.api as sm
from scipy import interpolate


class Stages(Enum):
    TimeSynchro = 0  # синхронизация каналов по времени
    Calibration = 1  # калибровка каналов
    Measurement1 = 2  # проведение эксперимента 1
    Measurement2 = 3  # проведение эксперимента 2


class Series:
    def __init__(self):

        self.time: np.ndarray = None   # отсчеты времени
        self.data: np.ndarray = None   # значения

        self.time_backup: np.ndarray = None
        self.data_backup: np.ndarray = None

        self.mask: bool = True         # применять обработку
        self.n_interp: Union[int, None] = 1024      # число точек интерполяции при загрузке из файла

        self.filtered = False

    @staticmethod
    def interpolate(time: np.asarray, data: np.asarray, n_interp: int = None):
        print('Series :: interpolate() | n_interp = {}'.format(n_interp))
        if n_interp is None:
            return time, data
        new_time = np.linspace(time[0], time[-1], n_interp)
        data = np.interp(new_time, time, data)
        return new_time, data

    def save_backup(self):
        print('Series :: save_backup()')
        self.time_backup = self.time
        self.data_backup = self.data

    def load_backup(self):
        print('Series :: load_backup()')
        self.time = self.time_backup
        self.data = self.data_backup


# noinspection PyTypeChecker
class Channel:
    def __init__(self, wavelength: float,
                 board: int = None, number: int = None,
                 used: bool = True, current_stage: Stages = Stages.Measurement2):
        self.log = Log()

        self.used = used  # использовать канал?
        self.current_stage = current_stage  # текущий этап

        self.board = board  # номер платы
        self.number = number  # номер канала
        self.wavelength = wavelength  # длина волны

        # 0 - TimeSynchro Series, 1 - Calibration Series, 2 - Measurement1 Series, 3 - Measurement2 Series
        self.Series = [Series() for _ in range(4)]
        # 0 - данные сеанса для определения временных сдвигов
        # 1 - данные калибровочного сеанса
        # 2 - данные эксперимента 1
        # 3 - данные эксперимента 2

        self.gain = 1.  # коэффициент усиления
        self.alpha = 1.  # калибровочный коэффициент, переводящий уровни с АЦП в интенсивности АЧТ
        self.timedelta = 0.  # смещение по времени

    @property
    def series(self) -> Series:
        return self.Series[self.current_stage.value]

    @property
    def filtered(self) -> bool:
        return self.Series[self.current_stage.value].filtered

    @filtered.setter
    def filtered(self, _val: bool):
        self.log.print("Channel #{} :: @filtered.setter".format(self.number))
        self.Series[self.current_stage.value].filtered = _val

    @property
    def data(self) -> np.ndarray:
        return self.series.data

    @data.setter
    def data(self, _data: np.ndarray):
        self.log.print("Channel #{} :: @data.setter".format(self.number))
        self.Series[self.current_stage.value].data = _data

    @property
    def data_backup(self) -> np.ndarray:
        return self.series.data_backup

    @data_backup.setter
    def data_backup(self, _data: np.ndarray):
        self.log.print("Channel #{} :: @data_backup.setter".format(self.number))
        self.Series[self.current_stage.value].data_backup = _data

    @property
    def time(self) -> np.ndarray:
        return self.series.time

    @time.setter
    def time(self, _time: np.ndarray):
        self.log.print("Channel #{} :: @time.setter".format(self.number))
        self.Series[self.current_stage.value].time = _time

    @property
    def time_backup(self) -> np.ndarray:
        return self.series.time_backup

    @time_backup.setter
    def time_backup(self, _time: np.ndarray):
        self.log.print("Channel #{} :: @time_backup.setter".format(self.number))
        self.Series[self.current_stage.value].time_backup = _time

    @property
    def data_gained(self) -> np.ndarray:
        return self.gain * self.data

    @property
    def data_calibrated(self) -> np.ndarray:  # калиброванные данные
        return self.alpha * self.gain * self.data

    @property
    def time_synced(self) -> np.ndarray:  # ход времени с учетом смещения
        return self.time - self.timedelta

    def find_peak(self, wavefront=False) -> tuple:
        self.log.print("Channel #{} :: find_peak()".format(self.number))
        peak = np.argmax(self.data)
        if wavefront:
            peak = np.argmax(np.diff(self.data)) - 1
        return self.time[peak], self.data[peak]

    @property
    def mask(self) -> bool:
        return self.series.mask

    @mask.setter
    def mask(self, _mask: bool):
        self.log.print("Channel #{} :: @mask.setter".format(self.number))
        self.Series[self.current_stage.value].mask = _mask

    @property
    def n_interp(self) -> Union[int, None]:
        return self.series.n_interp

    @n_interp.setter
    def n_interp(self, _val: Union[int, None]):
        self.log.print("Channel #{} :: @n_interp.setter".format(self.number))
        self.Series[self.current_stage.value].n_interp = _val

    def erase_current(self) -> None:
        self.log.print("Channel #{} :: erase_current()".format(self.number))
        self.Series[self.current_stage.value] = Series()

    def clear_all(self) -> None:
        self.log.print("Channel #{} :: clear_all()".format(self.number))
        self.Series = [Series() for _ in range(3)]

    def read(self, filepath: str, format_='txt'):
        self.log.print("Channel #{} :: read()".format(self.number))
        match format_:
            case 'txt':
                self.read_txt(filepath)
            case 'wfm':
                self.read_wfm(filepath)
            case _:
                self.read_txt(filepath)

    def read_txt(self, filepath: str):
        self.log.print("Channel #{} :: read_txt() | n_interp = {}".format(self.number, self.n_interp))
        series = np.asarray(np.loadtxt(filepath))
        if series.shape != (len(series), 2):
            raise "Неверный формат данных"

        self.time, self.data = \
            Series.interpolate(time=series[:, 0], data=series[:, 1], n_interp=self.n_interp)

    def read_wfm(self, filepath: str, frameNo=0):
        self.log.print("Channel #{} :: read_wfm()".format(self.number))
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
    def __init__(self, channels: List[Channel] = None, logger: Log = None):
        if logger is None:
            logger = Log()
        self.log = logger

        self.log.print('Session :: __init__()')
        self.channels: List[Channel] = channels  # Каналы

        # Настройка коэффициента усиления
        self.T_gain_cal = 2500                        # температура эталона

        # Фильтрация
        self.filter_params = initials.filter_parameters

        # Режим синхронизации
        self.wavefront_mode = False

        # Калибровка
        # Абсолютная калибровка
        self.T_abs_cal = 2500                         # температура эталона
        self.eps_abs_cal = 1                          # излучательная способность эталона
        # Относительная калибровка
        self.T_rel_cal = 2500                         # температура эталона
        self.eps_rel_cal = 1                          # излучательная способность эталона

        # Измерение образца
        self.eps_sample = 1.                          # излучательная способность образца (эксперимент 1)
        self.tp_n_interp_exp1: Union[int, None] = 100  # число точек интерполяции при расчете эксперимента 1
        self.tp_n_interp_exp2: Union[int, None] = 100  # число точек интерполяции при расчете эксперимента 2
        self.T_bounds: Tuple[float, float] = (1e1, 1e4)  # ограничения по значению температуры (эксперимент 2)
        self.eps_bounds: Tuple[float, float] = (0, 1)  # ограничения по значению коэффициента излучения (эксперимент 2)

        # Кол-во ядер
        self.n_workers_exp1 = 8
        self.n_workers_exp2 = 8

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
    def stage(self, _stage: Stages = Stages.Measurement2):
        self.log.print("Session :: @stage.setter")
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
        self.log.print("Session :: @used_indexes.setter")
        for i in range(len(self.channels)):
            if i in indexes:
                self.channels[i].used = True
            else:
                self.channels[i].used = False
                self.channels[i].Series[Stages.TimeSynchro.value].mask = False
                self.channels[i].Series[Stages.Calibration.value].mask = False
                self.channels[i].Series[Stages.Measurement1.value].mask = False
                self.channels[i].Series[Stages.Measurement2.value].mask = False

    @property
    def used(self) -> np.ndarray:
        arr = []
        for i in range(len(self.channels)):
            arr.append(i in self.used_indexes)
        return np.asarray(arr)

    @used.setter
    def used(self, _mask):
        self.log.print("Session :: @used.setter")
        for i in range(len(self.channels)):
            self.channels[i].used = _mask[i]

    @property
    def channels_used(self) -> list:
        return [self.channels[i] for i in self.used_indexes]

    @property
    def ready_indexes(self) -> list:
        indexes = []
        for i in self.used_indexes:
            if isinstance(self.channels[i].time, np.ndarray) and isinstance(self.channels[i].data, np.ndarray):
                indexes.append(i)
        return indexes

    @property
    def ready(self) -> np.ndarray:
        arr = []
        for i in range(len(self.channels)):
            arr.append(i in self.ready_indexes)
        return np.asarray(arr)

    @property
    def channels_ready(self) -> list:
        return [self.channels[i] for i in self.ready_indexes]

    @property
    def mask_indexes(self) -> list:
        mask_ = []
        for i in self.used_indexes:
            if self.channels[i].mask:
                mask_.append(i)
        return mask_

    @mask_indexes.setter
    def mask_indexes(self, indexes):
        self.log.print("Session :: @mask_indexes.setter")
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
        self.log.print("Session :: @mask.setter")
        for i in range(len(self.channels)):
            self.channels[i].used = (_mask[i] and (i in self.used_indexes))

    @property
    def channels_masked(self) -> list:
        return [self.channels[i] for i in self.mask_indexes]

    @property
    def wavelengths_masked(self) -> np.ndarray:
        return np.asarray([self.channels[i].wavelength for i in self.mask_indexes])

    @property
    def valid_indexes(self) -> list:
        indexes = []
        for i, channel in enumerate(self.channels):
            if isinstance(channel.time, np.ndarray) and \
                    isinstance(channel.data, np.ndarray) and \
                    (i in self.mask_indexes):
                indexes.append(i)
        return indexes

    @property
    def valid(self) -> np.ndarray:
        arr = []
        for i in range(len(self.channels)):
            arr.append(i in self.valid_indexes)
        return np.asarray(arr)

    @property
    def channels_valid(self) -> list:
        return [self.channels[i] for i in self.valid_indexes]

    @property
    def wavelengths_valid(self) -> np.ndarray:
        return np.asarray([self.channels[i].wavelength for i in self.valid_indexes])

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
        self.log.print("Session :: read_channels()")
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
                    self.channels[k + i].number = k + i + 1
                    self.channels[k + i].time, self.channels[k + i].data = \
                        Series.interpolate(series[:, 0], series[:, i + 1], self.channels[k + i].n_interp)
                    self.channels[k + i].mask = True
                    self.channels[k + i].log = self.log
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
        self.log.print("Session :: read_channel()")
        self.channels[index].number = index + 1
        self.channels[index].read(filepath=filepath, format_=format_)
        self.channels[index].mask = True
        self.channels[index].log = self.log

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
        self.log.print("Session :: set_gain()")
        for i, index in enumerate(self.ready_indexes):
            if values is None:
                goal_level = Session.__goal_level(wavelength=self.channels[index].wavelength, T=self.T_gain_cal)
                self.channels[index].gain = goal_level / np.mean(self.channels[index].data)
            else:
                self.channels[index].gain = values[i]

    def absolute_calibration(self):
        """
        Абсолютная калибровка
        """
        self.log.print("Session :: absolute_calibration()")
        body = Body(eps=self.eps_abs_cal)
        for index in self.valid_indexes:
            goal_intensity = body.intensity(wavelength=self.channels[index].wavelength, T=self.T_abs_cal)
            self.channels[index].alpha = goal_intensity / np.mean(self.channels[index].data_gained)

    # noinspection PyTypeChecker
    def relative_calibration(self):
        """
        Относительная калибровка
        """
        self.log.print("Session :: relative_calibration()")
        body = Body(eps=self.eps_rel_cal)
        I_spectrum = [body.intensity(wavelength=wavelength, T=self.T_rel_cal) for wavelength in self.wavelengths_valid]
        i_max = np.argmax(I_spectrum)
        index_max = self.valid_indexes[i_max]
        for i, index in enumerate(self.valid_indexes):
            self.channels[index].alpha = I_spectrum[i] / I_spectrum[i_max] * \
                np.mean(self.channels[index_max].data_gained) / np.mean(self.channels[index].data_gained)

    def set_timedelta(self):
        """
        Вычислить смещения каналов по времени
        """
        self.log.print("Session :: set_timedelta()")

        if len(self.valid_indexes) == 0:
            return

        first = self.valid_indexes[0]
        t0, _ = self.channels[first].find_peak(self.wavefront_mode)
        self.channels[first].timedelta = 0.
        for i in self.valid_indexes[1:]:
            t1, _ = self.channels[i].find_peak(self.wavefront_mode)
            self.channels[i].timedelta = t1 - t0

    def get_bounds(self, t_start: float = None, t_stop: float = None, mode='valid'):
        self.log.print("Session :: get_bounds()")
        indexes = []
        if mode == 'valid':
            indexes = self.valid_indexes
        elif mode == 'ready':
            indexes = self.ready_indexes

        bounds = np.asarray([(self.channels[i].time_synced[0], self.channels[i].time_synced[-1])
                             for i in indexes])
        left, right = np.max(bounds[:, 0]), np.min(bounds[:, 1])

        if t_start is None:
            t_start = left
        if t_stop is None:
            t_stop = right
        return t_start, t_stop

    def apply_filter(self, filter_name: str, t_start: float = None, t_stop: float = None):
        self.log.print("Session :: apply_filter() | filter name is {}".format(filter_name))
        t_start, t_stop = self.get_bounds(t_start, t_stop, mode='ready')

        for i in self.ready_indexes:
            if not self.channels[i].filtered:
                self.channels[i].time_backup = self.channels[i].time
                self.channels[i].data_backup = self.channels[i].data

            cond = (t_start <= self.channels[i].time) & (self.channels[i].time <= t_stop)
            time = self.channels[i].time[cond]
            data = self.channels[i].data[cond]

            match filter_name:
                case 'convolve':
                    box_pts = self.filter_params['convolve']['length']
                    box = np.ones(box_pts) / box_pts
                    data = np.convolve(data, box, mode='same')
                case 'fft':
                    w = fft.rfft(data)
                    spectrum = w**2
                    cutoff_idx = spectrum < (spectrum.max() / self.filter_params['fft']['divider'])
                    w2 = w.copy()
                    w2[cutoff_idx] = 0
                    data = fft.irfft(w2)
                case 'savgol':
                    data = savgol_filter(data,
                                         window_length=self.filter_params['savgol']['length'],
                                         polyorder=self.filter_params['savgol']['polyorder'])
                case 'uniform':
                    data = uniform_filter1d(data, size=self.filter_params['uniform']['length'])
                case 'lowess':
                    frac = self.filter_params['lowess']['fraction']
                    if frac > 1.:
                        frac = 1.
                    data = sm.nonparametric.lowess(data, time, frac=frac, is_sorted=True, return_sorted=False)
                case 'gaussian':
                    sigma = self.filter_params['gaussian']['sigma']
                    data = gaussian_filter(data, sigma=sigma)
                case 'spline':
                    spline = interpolate.UnivariateSpline(time, data)
                    spline.set_smoothing_factor(self.filter_params['spline']['smoothing'])
                    data = spline(time)
            self.channels[i].time = time
            self.channels[i].data = data
            self.channels[i].filtered = True

    def remove_filters(self):
        self.log.print("Session :: remove_filters()")
        for i in self.ready_indexes:
            if self.channels[i].filtered:
                self.channels[i].time = self.channels[i].time_backup
                self.channels[i].data = self.channels[i].data_backup
                self.channels[i].filtered = False

    def process_temperature(self, i):
        return i, self.sample.temperature(wavelengths=self.wavelengths_valid, intensities=self.__data[i])

    def get_temperature(self, t_start: float = None, t_stop: float = None, parallel: bool = True):
        """
        Расчет температуры по Планку
        """
        self.log.print("Session :: get_temperature()")
        t_start, t_stop = self.get_bounds(t_start, t_stop, mode='valid')

        data = []
        lengths = []
        for i in self.valid_indexes:
            cond = (t_start <= self.channels[i].time_synced) & (self.channels[i].time_synced <= t_stop)
            lengths.append(np.count_nonzero(cond))
            data.append([self.channels[i].time_synced[cond], self.channels[i].data_calibrated[cond]])

        n_interp = self.tp_n_interp_exp1
        if n_interp is None:
            n_interp = np.min(lengths)

        time = np.linspace(t_start, t_stop, n_interp)
        for i, _ in enumerate(self.valid_indexes):
            data[i] = np.interp(time, data[i][0], data[i][1])
        self.__data = np.asarray(data).T

        if not parallel:
            self.log.print('PARALLEL = FALSE')
            T = []
            for i, spectrum in enumerate(self.__data):
                T.append(self.sample.temperature(wavelengths=self.wavelengths_valid, intensities=spectrum))
                print('\rВыполнено\t{:.2f} %'.format((i + 1) / len(self.__data) * 100.), flush=True, end='          ')
            print('\n')
            return time, np.asarray(T)

        self.log.print('PARALLEL PROCESSING')
        results = []
        n = len(self.__data)
        with Pool(processes=self.n_workers_exp1) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(self.process_temperature, range(n)), total=n):
                results.append(result)

        results = np.asarray(sorted(results, key=lambda e: e[0]))
        return time, results[:, 1]  # T

    def process_temperature_and_emissivity(self, i):
        T, eps = self.sample.temperature_and_emissivity(wavelengths=self.wavelengths_valid, intensities=self.__data[i],
                                                        T_bounds=self.T_bounds, eps_bounds=self.eps_bounds)
        return i, T, eps

    def get_temperature_and_emissivity(self, t_start: float = None, t_stop: float = None, parallel: bool = True):
        """
        Подбор температуры и излучательной способности по Планку
        """
        self.log.print("Session :: get_temperature_and_emissivity()")
        t_start, t_stop = self.get_bounds(t_start, t_stop, mode='valid')

        data = []
        lengths = []
        for i in self.valid_indexes:
            cond = (t_start <= self.channels[i].time_synced) & (self.channels[i].time_synced <= t_stop)
            lengths.append(np.count_nonzero(cond))
            data.append([self.channels[i].time_synced[cond], self.channels[i].data_calibrated[cond]])

        n_interp = self.tp_n_interp_exp2
        if n_interp is None:
            n_interp = np.min(lengths)

        time = np.linspace(t_start, t_stop, n_interp)
        for i, _ in enumerate(self.valid_indexes):
            data[i] = np.interp(time, data[i][0], data[i][1])
        self.__data = np.asarray(data).T

        if not parallel:
            self.log.print('PARALLEL = FALSE')
            T, eps = [], []
            for i, spectrum in enumerate(self.__data):
                temperature, epsilon = self.sample.temperature_and_emissivity(wavelengths=self.wavelengths_valid,
                                                                              intensities=spectrum,
                                                                              T_bounds=self.T_bounds,
                                                                              eps_bounds=self.eps_bounds)
                T.append(temperature)
                eps.append(epsilon)
                print('\rВыполнено\t{:.2f} %'.format((i + 1) / len(self.__data) * 100.), flush=True, end='          ')
            print('\n')
            return time, np.asarray(T), np.asarray(eps)

        self.log.print('PARALLEL PROCESSING')
        results = []
        n = len(self.__data)
        with Pool(processes=self.n_workers_exp2) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(self.process_temperature_and_emissivity, range(n)), total=n):
                results.append(result)

        results = np.asarray(sorted(results, key=lambda e: e[0]))
        return time, results[:, 1], results[:, 2]  # T and eps

    def save(self, filepath='session') -> None:
        """
        Сохранить сессию
        """
        self.log.print("Session :: save()")
        info = {'channels': []}

        for attr_name in ['T_gain_cal', 'T_abs_cal', 'eps_abs_cal', 'T_rel_cal', 'eps_rel_cal',
                          'eps_sample', 'tp_n_interp_exp1', 'tp_n_interp_exp2', 'n_workers_exp1', 'n_workers_exp2',
                          'T_bounds', 'eps_bounds']:
            info[attr_name] = getattr(self, attr_name)

        for channel in self.channels:
            fields = {'series': [],
                      'current_stage': channel.current_stage.value}

            for attr_name in ['used', 'board', 'number', 'wavelength', 'gain', 'alpha', 'timedelta']:
                fields[attr_name] = getattr(channel, attr_name)

            for series in channel.Series:
                fields_series = {}
                for attr_name in ['time', 'data']:
                    field = getattr(series, attr_name)
                    if isinstance(field, np.ndarray):
                        field = field.tolist()
                    fields_series[attr_name] = field
                for attr_name in ['mask', 'n_interp']:
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
        print("Session :: load()")
        with open(filepath, 'r') as dump:
            info = json.load(dump)

        session = Session()

        for attr_name in ['T_gain_cal', 'T_abs_cal', 'eps_abs_cal', 'T_rel_cal', 'eps_rel_cal',
                          'eps_sample', 'tp_n_interp_exp1', 'tp_n_interp_exp2', 'n_workers_exp1', 'n_workers_exp2',
                          'T_bounds', 'eps_bounds']:
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
                for attr_name in ['time', 'data']:
                    val = info['channels'][i]['series'][j][attr_name]
                    if isinstance(val, list):
                        val = np.array(val)
                    setattr(series, attr_name, val)
                for attr_name in ['mask', 'n_interp']:
                    setattr(series, attr_name, info['channels'][i]['series'][j][attr_name])
                _Series.append(series)

            Channels[i].Series = _Series

        session.channels = Channels
        return session

    def clear(self):
        """
        Очистить сессию
        """
        self.log.print("Session :: clear()")
        self.__init__()
        self.save()

    def erase(self):
        """
        Очистить данные каналов
        """
        self.log.print("Session :: erase()")
        for channel in self.channels:
            channel.erase_current()

    @property
    def has_valid_channels(self):
        return len(self.valid_indexes) > 0


if __name__ == "__main__":
    import core.test
    core.test.session_001()
