from typing import Union, Tuple
from functools import partial
from sys import exit
import traceback

from formation import AppBuilder

from tkinter import Variable
from tkinter import Toplevel
from tkinter import filedialog, messagebox
from tkinter import Label, Spinbox, Button
from tkinter import NORMAL, DISABLED
from tkinter import TclError

import core.initials as initials
from core.session import Session, Channel, Stages
from core.logger import Log

import re
import numpy as np
import pandas as pd
import openpyxl

from matplotlib import pyplot as plt
from matplotlib import cm


# MENU Файл
def new_session(event=None):
    global session, session_filepath

    log.print('Globals :: new_session()')

    session = Session(channels=channels_initial)
    session_filepath = None

    update_interface()


def open_session(event=None):
    global session, session_filepath

    log.print('Globals :: open_session()')

    filepath = filedialog.askopenfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])

    if filepath:
        session = Session.load(filepath)
        session_filepath = filepath
        update_interface()


def save_session(event=None):
    log.print('Globals :: save_session()')
    if session_filepath:
        update_session()
        session.save(session_filepath)
    else:
        save_as_session()


def save_as_session(event=None):
    log.print('Globals :: save_as_session()')
    global session_filepath
    update_session()
    session_filepath = filedialog.asksaveasfilename(filetypes=[('Параметры сессии (JSON)', '*.json')], defaultextension='.json')
    session.save(session_filepath)


def on_close(event=None):
    log.print('Globals :: on_close()')
    # session.save('session.json')
    log.print("До свидания!")
    exit(0)


# MENU Инструменты
# noinspection PyBroadException
def set_gain(event=None, T=2500):
    log.print('Globals :: set_gain() | T = {}'.format(T))
    global session
    session.T_gain_cal = T
    try:
        session.set_gain()
    except TypeError as e:
        messagebox.showerror(title='Ошибка', message='Невозможно рассчитать коэффициенты усиления. \n Данные сеанса неверные либо отсутствуют')
        log.print_exception(e)
    except Exception as e:
        messagebox.showerror(title='Ошибка', message=str(e))
        log.print_exception(e)
    finally:
        update_interface()
        root.notebook_main.select(Stages.Calibration.value + 1)


def apply_filter(event=None, name='fft'):
    global session
    global t_start, t_stop

    log.print('Globals :: apply_filter() | name = {}'.format(name))

    session.apply_filter(filter_name=name, t_start=t_start, t_stop=t_stop)

    plot()


class FilterParametersDialog:
    def __init__(self, parent):
        log.print('FilterParametersDialog :: __init__()')

        self.filter_parameters = initials.filter_parameters

        top = self.top = Toplevel(parent)
        top.title('Параметры фильтров')
        top.geometry('{:.0f}x{:.0f}'.format(720, 400))
        top.resizable(False, False)

        kwargs = {'columnspan': 2, 'padx': 10, 'pady': 8}
        Label(top, text='Фильтр', font='-weight bold', justify='left', width=20).grid(row=0, column=0, **kwargs)
        Label(top, text='Параметр', font='-weight bold', justify='left', width=20).grid(row=0, column=2, **kwargs)
        Label(top, text='Значение', font='-weight bold', justify='left', width=20).grid(row=0, column=4, **kwargs)

        Label(top, text='Сверточный', justify='left', width=20).grid(row=2, column=0, **kwargs)
        Label(top, text='Число точек', justify='left', width=20).grid(row=2, column=2, **kwargs)
        self.convolve_length = Variable(top, value=3)
        Spinbox(top, from_=3, to=20, increment=1, textvariable=self.convolve_length, width=10).grid(row=2, column=4, **kwargs)

        Label(top, text='БПФ-сглаживание', justify='left', width=20).grid(row=3, column=0, **kwargs)
        Label(top, text='Делитель', justify='left', width=20).grid(row=3, column=2, **kwargs)
        self.fft_divider = Variable(top, value=5)
        Spinbox(top, from_=3, to=20, increment=1, textvariable=self.fft_divider, width=10).grid(row=3, column=4, **kwargs)

        Label(top, text='Savgol', justify='left', width=20).grid(row=4, column=0, **kwargs)
        Label(top, text='Длина окна', justify='left', width=20).grid(row=4, column=2, **kwargs)
        self.savgol_length = Variable(top, value=11)
        Spinbox(top, from_=5, to=20, increment=1, textvariable=self.savgol_length, width=10).grid(row=4, column=4, **kwargs)
        Label(top, text='Порядок полинома', justify='left', width=20).grid(row=5, column=2, **kwargs)
        self.savgol_polyorder = Variable(top, value=3)
        Spinbox(top, from_=3, to=9, increment=1, textvariable=self.savgol_polyorder, width=10).grid(row=5, column=4, **kwargs)
        
        Label(top, text='Однородный', justify='left', width=20).grid(row=6, column=0, **kwargs)
        Label(top, text='Длина', justify='left', width=20).grid(row=6, column=2, **kwargs)
        self.uniform_length = Variable(top, value=5)
        Spinbox(top, from_=3, to=20, increment=1, textvariable=self.uniform_length, width=10).grid(row=6, column=4, **kwargs)
        
        Label(top, text='LOWESS', justify='left', width=20).grid(row=7, column=0, **kwargs)
        Label(top, text='Отношение', justify='left', width=20).grid(row=7, column=2, **kwargs)
        self.lowess_fraction = Variable(top, value=0.35)
        Spinbox(top, from_=0.01, to=20, increment=0.01, textvariable=self.lowess_fraction, width=10).grid(row=7, column=4, **kwargs)

        Label(top, text='Гауссиан', justify='left', width=20).grid(row=8, column=0, **kwargs)
        Label(top, text='Сигма', justify='left', width=20).grid(row=8, column=2, **kwargs)
        self.gaussian_sigma = Variable(top, value=5)
        Spinbox(top, from_=3, to=20, increment=1, textvariable=self.gaussian_sigma, width=10).grid(row=8, column=4, **kwargs)

        Label(top, text='Сплайны', justify='left', width=20).grid(row=9, column=0, **kwargs)
        Label(top, text='Сглаживание', justify='left', width=20).grid(row=9, column=2, **kwargs)
        self.spline_smoothing = Variable(top, value=0.01)
        Spinbox(top, from_=0.01, to=20, increment=0.01, textvariable=self.spline_smoothing, width=10).grid(row=9, column=4, **kwargs)

        Button(top, text="Применить", command=self.set, width=40).grid(row=11, column=0, columnspan=6)

    def set(self):
        log.print('FilterParametersDialog :: set()')
        # filter_parameters = initials.filter_parameters
        parameters = {
            'convolve': {'length': self.convolve_length.get()},
            'fft': {'divider': self.fft_divider.get()},
            'savgol': {'length': self.savgol_length.get(), 'polyorder': self.savgol_polyorder.get()},
            'uniform': {'length': self.uniform_length.get()},
            'lowess': {'fraction': self.lowess_fraction.get()},
            'gaussian': {'sigma': self.gaussian_sigma.get()},
            'spline': {'smoothing': self.spline_smoothing.get()},
        }
        self.filter_parameters = parameters
        self.top.destroy()


def filter_parameters(event=None):
    global root, session

    log.print('Globals :: filter_parameters()')

    inputDialog = FilterParametersDialog(root._app)
    root._app.wait_window(inputDialog.top)
    session.filter_params = inputDialog.filter_parameters


def filter_clear(event=None):
    global session

    log.print('Globals :: filter_parameters()')

    session.remove_filters()


# Работа с фалами
def load_channel(event=None, channel=0, stage=Stages.Measurement2):
    global session, root

    log.print('Globals :: load_channel() | channel {} | stage {}'.format(channel, stage))

    update_session()

    index = channel
    session.stage = stage

    filepath = filedialog.askopenfilename(filetypes=[('Текстовый файл', '*.txt'), ('Файл WFM', '*.wfm')])
    if not filepath:
        return

    try:
        ext = re.split(r'\.', filepath)
        ext = ext[-1]
    except TypeError as e:
        log.print_exception(e)
        return

    # noinspection PyBroadException
    try:
        session.read_channel(index=index, filepath=filepath, format_=ext)

        match stage.value:
            case Stages.TimeSynchro.value:
                sync_load_buttons[index].configure(background="lightgreen")
            case Stages.Calibration.value:
                cal_load_buttons[index].configure(background="lightgreen")
            case Stages.Measurement1.value:
                exp1_load_buttons[index].configure(background="lightgreen")
            case Stages.Measurement2.value:
                exp2_load_buttons[index].configure(background="lightgreen")

    except Exception as e:
        messagebox.showerror(title='Ошибка чтения файла', message=str(e))
        log.print_exception(e)
        match stage.value:
            case Stages.TimeSynchro.value:
                sync_load_buttons[index].configure(background="pink")
            case Stages.Calibration.value:
                cal_load_buttons[index].configure(background="pink")
            case Stages.Measurement1.value:
                exp1_load_buttons[index].configure(background="pink")
            case Stages.Measurement2.value:
                exp2_load_buttons[index].configure(background="pink")


def load_all(event=None, stage=Stages.Measurement2):
    global session, root

    log.print('Globals :: load_all()')

    update_session()

    session.stage = stage

    filepaths = filedialog.askopenfilenames(filetypes=[('Файл CSV', '*.csv'), ('Файл DAT', '*.dat'),
                                                       ('Текстовый файл', '*.txt'), ('Файл WFM', '*.wfm')])
    if not filepaths:
        return

    if len(filepaths) > 10:
        messagebox.showwarning(title='Предупреждение', message='Вы пытаетесь загрузить больше 10 файлов')
        filepaths = filepaths[:10]

    extensions = []
    try:
        for filepath in filepaths:
            ext = re.split(r'\.', filepath)
            extensions.append(ext[-1])
    except TypeError as e:
        log.print_exception(e)
        return

    if len(np.unique(extensions)) > 1:
        messagebox.showerror(title='Ошибка', message="Пожалуйста, выберите файлы одного и того же типа")

    ext = np.unique(extensions)[0]

    # noinspection PyBroadException
    try:
        session.read_channels(filepaths=filepaths, format_=ext)
    except IndexError as e:
        messagebox.showwarning(title='Предупреждение', message='Вы пытаетесь загрузить больше 10 каналов')
        log.print_exception(e)
    except Exception as e:
        messagebox.showerror(title='Ошибка чтения файла', message=str(e))
        log.print_exception(e)

    update_interface()


def clear_all(event=None, stage=Stages.Measurement2):
    global session, root

    log.print('Globals :: clear_all()')

    session.erase()
    update_interface()


# Визуализация
# noinspection PyUnresolvedReferences
def plt_on_button_press(event):
    global area
    global t_start

    log.print('Globals :: plt_on_button_press()')

    if area is not None:
        area.remove()
        area = None
    match event.button:
        case 1:
            t_start = event.xdata
        case 3:
            t_start = None


# noinspection PyUnresolvedReferences,PyTypeChecker
def plt_on_button_release(event):
    global figure, axes, area
    global t_start, t_stop

    log.print('Globals :: plt_on_button_release()')

    match event.button:
        case 1:
            t_stop = event.xdata
            y_min, y_max = axes.get_ylim()
            area = axes.fill_between(x=np.linspace(t_start, t_stop, 10),
                                     y1=[y_max] * 10, y2=[y_min] * 10,
                                     color='blue', alpha=0.3)
            axes.set_ylim(y_min, y_max)
        case 3:
            t_stop = None

    figure.canvas.draw()
    figure.canvas.flush_events()


def show(event=None, stage=Stages.Measurement2):  # показать исходные данные
    global root
    global figure, axes

    log.print('Globals :: show() | stage = {}'.format(stage))

    update_session()

    fig, ax = plt.subplots()

    figure, axes = fig, ax

    if root.set_time_interval.get():
        fig.canvas.mpl_connect('button_press_event', plt_on_button_press)
        fig.canvas.mpl_connect('button_release_event', plt_on_button_release)

    for channel in session.channels_ready:
        ax.plot(channel.time, channel.data,
                label='Канал #{}'.format(channel.number))

    ax.set_xlabel('Время')
    ax.set_ylabel('Сигнал')

    ax.legend(loc='best', frameon=True)

    ax.grid(ls=':', color='gray')

    plt.tight_layout()
    plt.show()


def plot(event=None, stage=Stages.Measurement2):  # показать результат обработки
    global result

    log.print('Globals :: plot() | stage = {}'.format(stage))

    update_session()
    fig, ax = plt.subplots()
    if stage.value != Stages.Measurement2.value:
        for j, channel in enumerate(session.channels_valid):
            ax.plot(channel.time_synced, channel.data_calibrated,
                    label='Канал #{}'.format(channel.number))
        ax.set_xlabel('Время')
        ax.set_ylabel('Сигнал')
        ax.legend(loc="upper right", frameon=True)
        ax.grid(ls=':', color='gray')

    if stage.value == Stages.Measurement1.value:
        if result is not None:
            time, T = result

            ax = plt.twinx(ax)
            ax.plot(time, T, color='blue', lw=2, label='Температура')
            ax.set_ylabel('Температура, К')
            ax.legend(loc="upper center", frameon=True)

    if stage.value == Stages.Measurement2.value:
        if result is not None:
            time, T, eps = result

            ax.plot(time, T, color='crimson', lw=2, label='Температура')
            ax.set_ylabel('Температура, К')
            ax.legend(loc="upper left", frameon=True)

            ax = plt.twinx(ax)
            ax.plot(time, eps, color='blue', lw=2, label='Коэфф. излучения')
            ax.set_ylabel('Излучательная способность, безразм.')
            ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.draw()

    if root.mode_3D_exp1.get() and \
            stage.value == Stages.Measurement1.value and \
            result is not None:
        time, T = result

        spectra = []
        for temperature in T:
            spectrum = session.sample.intensity(wavelength=session.wavelengths_valid, T=temperature)
            spectra.append(spectrum)

        X = np.asarray([time] * len(session.valid_indexes))
        Y = np.moveaxis(np.asarray([session.wavelengths_valid] * len(time)), 0, 1)
        Z = np.asarray(spectra).T

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.plot_wireframe(X, Y, Z, rstride=1, cstride=10)
        ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=False)
        ax.set_xlabel(r'Время')
        ax.set_ylabel(r'Длина волны (мкм)')
        ax.set_zlabel(r'Интенсивность (Вт/см$^2$/мкм)')

    plt.show()


# Вкладка "Синхронизация"
def apply_sync(event=None):
    global session, root

    log.print('Globals :: apply_sync()')

    update_session()

    session.set_timedelta()
    update_interface()


# Вкладка "Калибровка"
def set_eps_cal(event=None):  # Излучательная способность эталона
    global session, root

    log.print('Globals :: set_eps_cal()')

    eps = 1
    filepath = filedialog.askopenfilename(filetypes=[("Таблица Excel", ".xlsx .xls")])

    if filepath:
        # noinspection PyBroadException
        try:
            df = pd.read_excel(filepath, index_col=None, header=None, engine="openpyxl")
            eps = dict(df.to_numpy().tolist())
        except Exception as e:
            messagebox.showerror(title='Ошибка', message=str(e))
            log.print_exception(e)
            eps = 1

    if root.calibration_type.get():  # Относительная калибровка
        session.eps_rel_cal = eps
    else:  # Абсолютная калибровка
        session.eps_abs_cal = eps


def apply_cal(event=None):
    global session

    log.print('Globals :: apply_cal()')

    if root.calibration_type.get():  # Относительная калибровка
        session.relative_calibration()
    else:
        session.absolute_calibration()
    update_interface()


# Вкладка "Эксперимент"
def set_eps_exp(event=None):
    global session, root

    log.print('Globals :: set_eps_exp()')

    eps = 1
    filepath = filedialog.askopenfilename(filetypes=[("Таблица Excel", ".xlsx .xls")])

    if filepath:
        # noinspection PyBroadException
        try:
            df = pd.read_excel(filepath, index_col=None, header=None, engine="openpyxl")
            eps = dict(df.to_numpy().tolist())
        except Exception as e:
            messagebox.showerror(title='Ошибка', message=str(e))
            log.print_exception(e)
            eps = 1

    session.eps_sample = eps


def calculate_temperatures(event=None):
    global session

    log.print('Globals :: calculate_temperatures()')

    update_session()

    global t_start, t_stop
    global result

    try:
        parallel = root.parallelize_exp1.get()
        result = session.get_temperature(t_start=t_start, t_stop=t_stop, parallel=parallel)
    except Exception as e:
        messagebox.showerror('Ошибка расчета температуры', str(e))
        log.print_exception(e)


def calculate_temperatures_and_emissivity(event=None):
    global session

    log.print('Globals :: calculate_temperatures()')

    update_session()

    global t_start, t_stop
    global result

    try:
        parallel = root.parallelize_exp2.get()

        # temp_min = root.temp_min_exp2.get()
        # temp_max = root.temp_max_exp2.get()
        #
        # eps_min = root.eps_min_exp2.get()
        # eps_max = root.eps_max_exp2.get()
        #
        # T_bounds = (temp_min, temp_max)
        # eps_bounds = (eps_min, eps_max)
        # session.T_bounds = T_bounds
        # session.eps_bounds = eps_bounds

        result = session.get_temperature_and_emissivity(t_start=t_start, t_stop=t_stop, parallel=parallel)
    except Exception as e:
        messagebox.showerror('Ошибка расчета температуры / излучательной способности', str(e))
        log.print_exception(e)


def export_temperatures(event=None):
    global result

    log.print('Globals :: export_temperatures()')

    if result is not None:
        time, T = result

        data = []
        for j, (timestamp, temperature) in enumerate(zip(time, T)):
            data.append([j + 1, timestamp, temperature])
        data = np.asarray(data)

        data = pd.DataFrame(data=data[:, 1:], index=data[:, 0],
                            columns=["время", 'температура, K'])

        filepath = filedialog.asksaveasfilename(filetypes=[("Таблица Excel", ".xlsx .xls")], defaultextension='.xlsx')

        if filepath:
            data.to_excel(filepath)


def export_temperatures_and_emissivity(event=None):
    global result

    log.print('Globals :: export_temperatures()')

    if result is not None:
        time, T, eps = result

        data = []
        for j, (timestamp, temperature, emissivity) in enumerate(zip(time, T, eps)):
            data.append([j + 1, timestamp, temperature, emissivity])
        data = np.asarray(data)

        data = pd.DataFrame(data=data[:, 1:], index=data[:, 0],
                            columns=["время", 'температура, K', 'коэфф. излучения, безразм.'])

        filepath = filedialog.asksaveasfilename(filetypes=[("Таблица Excel", ".xlsx .xls")], defaultextension='.xlsx')

        if filepath:
            data.to_excel(filepath)


# Основное
def update_session(*args):
    global session

    log.print('Globals :: update_session()')

    # root._app.update()
    # root._app.update_idletasks()

    session.used = list(map(lambda var: var.get(), usage))

    if root.sync_method.get():  # Синхронизация по точке начала фронта
        session.wavefront_mode = True
    else:
        session.wavefront_mode = False

    if root.calibration_type.get():  # Относительная калибровка
        session.T_rel_cal = root.T_cal.get()
    else:  # Абсолютная калибровка
        session.T_abs_cal = root.T_cal.get()

    session.tp_n_interp_exp1 = None
    if root.tp_interp_exp1.get():
        session.tp_n_interp_exp1 = root.tp_n_interp_exp1.get()

    session.tp_n_interp_exp2 = None
    if root.tp_interp_exp2.get():
        session.tp_n_interp_exp2 = root.tp_n_interp_exp2.get()

    if root.parallelize_exp1.get():
        session.n_workers_exp1 = root.n_workers_exp1.get()

    if root.parallelize_exp2.get():
        session.n_workers_exp2 = root.n_workers_exp2.get()

    temp_min = root.temp_min_exp2.get()
    temp_max = root.temp_max_exp2.get()

    eps_min = root.eps_min_exp2.get()
    eps_max = root.eps_max_exp2.get()

    T_bounds = (temp_min, temp_max)
    eps_bounds = (eps_min, eps_max)
    session.T_bounds = T_bounds
    session.eps_bounds = eps_bounds

    for j in range(10):
        try:
            session.channels[j].wavelength = lambdas[j].get()
            session.channels[j].gain = gains[j].get()
            session.channels[j].timedelta = deltas[j].get()
            session.channels[j].alpha = alphas[j].get()

            for stage in Stages:
                n_interp = None
                mask = False
                match stage.value:
                    case Stages.TimeSynchro.value:
                        mask = sync_mask[j].get()
                        if root.fr_interp_sync.get():
                            n_interp = root.fr_n_interp_sync.get()
                    case Stages.Calibration.value:
                        mask = cal_mask[j].get()
                        if root.fr_interp_cal.get():
                            n_interp = root.fr_n_interp_cal.get()
                    case Stages.Measurement1.value:
                        mask = exp1_mask[j].get()
                        if root.fr_interp_exp1.get():
                            n_interp = root.fr_n_interp_exp1.get()
                    case Stages.Measurement2.value:
                        mask = exp2_mask[j].get()
                        if root.fr_interp_exp2.get():
                            n_interp = root.fr_n_interp_exp2.get()

                session.channels[j].Series[stage.value].n_interp = n_interp
                session.channels[j].Series[stage.value].mask = mask

        except TclError:
            pass

    tab_index = root.notebook_main.index(root.notebook_main.select())
    stage_val = tab_index - 1
    if 0 <= stage_val < 4:
        session.stage = Stages(stage_val)

    # update_interface()


def update_interface(*args):
    global root

    log.print('Globals :: update_interface()')

    for stage in Stages:
        n_interp = session.first_channel.Series[stage.value].n_interp
        match stage.value:
            case Stages.TimeSynchro.value:
                root.fr_interp_sync.set(bool(n_interp))
                root.fr_n_interp_sync.set(n_interp)
            case Stages.Calibration.value:
                root.fr_interp_cal.set(bool(n_interp))
                root.fr_n_interp_cal.set(n_interp)
            case Stages.Measurement1.value:
                root.fr_interp_exp1.set(bool(n_interp))
                root.fr_n_interp_exp1.set(n_interp)
            case Stages.Measurement2.value:
                root.fr_interp_exp2.set(bool(n_interp))
                root.fr_n_interp_exp2.set(n_interp)

    for j in range(10):

        if j in session.used_indexes:
            usage_checkbuttons[j].select()
        else:
            usage_checkbuttons[j].deselect()

        lambdas[j].set(session.channels[j].wavelength)

        gains[j].set(session.channels[j].gain)

        deltas[j].set(session.channels[j].timedelta)

        alphas[j].set(session.channels[j].alpha)

        for stage in Stages:
            mask = session.channels[j].Series[stage.value].mask
            loaded = isinstance(session.channels[j].Series[stage.value].data, np.ndarray)
            match stage.value:
                case Stages.TimeSynchro.value:
                    sync_mask[j].set(mask)
                    sync_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)
                case Stages.Calibration.value:
                    cal_mask[j].set(mask)
                    cal_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)
                case Stages.Measurement1.value:
                    exp1_mask[j].set(mask)
                    exp1_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)
                case Stages.Measurement2.value:
                    exp2_mask[j].set(mask)
                    exp2_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)

    root.tp_interp_exp1.set(bool(session.tp_n_interp_exp1))
    root.tp_interp_exp2.set(bool(session.tp_n_interp_exp2))
    root.tp_n_interp_exp1.set(session.tp_n_interp_exp1)
    root.tp_n_interp_exp2.set(session.tp_n_interp_exp2)

    root.n_workers_exp1.set(session.n_workers_exp1)
    root.n_workers_exp2.set(session.n_workers_exp2)

    root.temp_min_exp2.set(session.T_bounds[0])
    root.temp_max_exp2.set(session.T_bounds[1])

    root.eps_min_exp2.set(session.eps_bounds[0])
    root.eps_max_exp2.set(session.eps_bounds[1])

    # root._app.update()
    # root._app.update_idletasks()

    current_stage = session.stage
    root.notebook_main.select(current_stage.value + 1)


def update_usage(*args):
    global session, root

    log.print('Globals :: update_usage()')

    for j in range(10):
        if usage[j].get():
            lambdas_spinboxes[j].configure(state=NORMAL)
            deltas_spinboxes[j].configure(state=NORMAL)
            alphas_spinboxes[j].configure(state=NORMAL)
            gains_spinboxes[j].configure(state=NORMAL)

            sync_checkbuttons[j].configure(state=NORMAL)
            cal_checkbuttons[j].configure(state=NORMAL)
            exp1_checkbuttons[j].configure(state=NORMAL)
            exp2_checkbuttons[j].configure(state=NORMAL)

            sync_load_buttons[j].configure(state=NORMAL)
            cal_load_buttons[j].configure(state=NORMAL)
            exp1_load_buttons[j].configure(state=NORMAL)
            exp2_load_buttons[j].configure(state=NORMAL)

        else:
            for stage in Stages:
                session.channels[j].Series[stage.value].mask = False
                match stage.value:
                    case Stages.TimeSynchro.value:
                        sync_mask[j].set(False)
                    case Stages.Calibration.value:
                        cal_mask[j].set(False)
                    case Stages.Measurement1.value:
                        exp1_mask[j].set(False)
                    case Stages.Measurement2.value:
                        exp2_mask[j].set(False)

            lambdas_spinboxes[j].configure(state=DISABLED)
            deltas_spinboxes[j].configure(state=DISABLED)
            alphas_spinboxes[j].configure(state=DISABLED)
            gains_spinboxes[j].configure(state=DISABLED)

            sync_checkbuttons[j].configure(state=DISABLED)
            cal_checkbuttons[j].configure(state=DISABLED)
            exp1_checkbuttons[j].configure(state=DISABLED)
            exp2_checkbuttons[j].configure(state=DISABLED)

            sync_load_buttons[j].configure(state=DISABLED)
            cal_load_buttons[j].configure(state=DISABLED)
            exp1_load_buttons[j].configure(state=DISABLED)
            exp2_load_buttons[j].configure(state=DISABLED)


if __name__ == "__main__":
    log = Log()
    log.print('Добро пожаловать!')

    root = AppBuilder(path="window.xml")

    # Сессия
    channels_initial = [Channel(wavelength=wavelength) for _, wavelength in initials.configuration]
    session: Session = Session(channels=channels_initial, logger=log)
    # session: Session = Session.load('session.json')
    session_filepath = None

    figure, axes, area = None, None, None
    t_start, t_stop = None, None  # временной интервал
    result: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], None] = None

    root.notebook_main.select(0)

    # Checkbuttons
    usage_checkbuttons, sync_checkbuttons, cal_checkbuttons, exp1_checkbuttons, exp2_checkbuttons = [], [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("usage_checkbuttons.append(root.checkbutton_channel_{})".format(chn))
        exec("sync_checkbuttons.append(root.checkbutton_ch{}_sync)".format(chn))
        exec("cal_checkbuttons.append(root.checkbutton_ch{}_cal)".format(chn))
        exec("exp1_checkbuttons.append(root.checkbutton_ch{}_calc_exp1)".format(chn))
        exec("exp2_checkbuttons.append(root.checkbutton_ch{}_calc_exp2)".format(chn))

    # Spinboxes
    lambdas_spinboxes, deltas_spinboxes, alphas_spinboxes, gains_spinboxes = \
        [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("lambdas_spinboxes.append(root.spinbox_lambda_{})".format(chn))
        exec("deltas_spinboxes.append(root.spinbox_sync_{})".format(chn))
        exec("alphas_spinboxes.append(root.spinbox_alpha_{})".format(chn))
        exec("gains_spinboxes.append(root.spinbox_gain_{})".format(chn))

    temp_min_spinbox = root.spinbox_temp_min_exp2
    temp_max_spinbox = root.spinbox_temp_max_exp2
    eps_min_spinbox = root.spinbox_eps_min_exp2
    eps_max_spinbox = root.spinbox_eps_max_exp2

    # Load buttons
    sync_load_buttons, cal_load_buttons, exp1_load_buttons, exp2_load_buttons = [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("sync_load_buttons.append(root.button_load_ch{}_sync)".format(chn))
        exec("cal_load_buttons.append(root.button_load_ch{}_cal)".format(chn))
        exec("exp1_load_buttons.append(root.button_load_ch{}_exp1)".format(chn))
        exec("exp2_load_buttons.append(root.button_load_ch{}_exp2)".format(chn))

    original_button_color = root.button_load_all_exp1.cget("background")

    # Основные переменные
    usage, lambdas, deltas, alphas, gains = [], [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("usage.append(root.use_ch{})".format(chn))
        exec("lambdas.append(root.lambda{})".format(chn))
        exec("deltas.append(root.deltat{})".format(chn))
        exec("alphas.append(root.alpha{})".format(chn))
        exec("gains.append(root.gain{})".format(chn))

    for i in range(10):
        usage[i].trace('w', update_usage)

    # Другие переменные
    sync_mask, cal_mask, exp1_mask, exp2_mask = [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("sync_mask.append(root.sync_ch{})".format(chn))  # Синхронизация
        exec("cal_mask.append(root.cal_ch{})".format(chn))    # Калибровка
        exec("exp1_mask.append(root.exp1_ch{})".format(chn))   # Эксперимент 1
        exec("exp2_mask.append(root.exp2_ch{})".format(chn))  # Эксперимент 2

    update_interface()

    set_gain_T1500_depr, set_gain_T1773_depr, set_gain_T2000_depr, set_gain_T2500_depr = \
        partial(set_gain, T=1500), partial(set_gain, T=1773), partial(set_gain, T=2000), partial(set_gain, T=2500)

    s = ''
    for i in range(10):
        chn = str(i + 1).zfill(2)
        s += 'load_ch{}_sync, load_ch{}_cal, load_ch{}_exp1, load_ch{}_exp2, '.format(chn, chn, chn, chn)
    s = s[:-2] + ' = '
    for i in range(10):
        s += 'partial(load_channel, channel={}, stage=Stages.TimeSynchro), '.format(i)
        s += 'partial(load_channel, channel={}, stage=Stages.Calibration), '.format(i)
        s += 'partial(load_channel, channel={}, stage=Stages.Measurement1), '.format(i)
        s += 'partial(load_channel, channel={}, stage=Stages.Measurement2), '.format(i)
    exec(s[:-2])

    load_all_sync, load_all_cal, load_all_exp1, load_all_exp2 = \
        partial(load_all, stage=Stages.TimeSynchro), \
        partial(load_all, stage=Stages.Calibration), \
        partial(load_all, stage=Stages.Measurement1), partial(load_all, stage=Stages.Measurement2)

    clear_all_sync, clear_all_cal, clear_all_exp1, clear_all_exp2 = \
        partial(clear_all, stage=Stages.TimeSynchro), \
        partial(clear_all, stage=Stages.Calibration), \
        partial(clear_all, stage=Stages.Measurement1), partial(clear_all, stage=Stages.Measurement2)

    show_sync, show_cal, show_exp1, show_exp2 = \
        partial(show, stage=Stages.TimeSynchro), \
        partial(show, stage=Stages.Calibration), \
        partial(show, stage=Stages.Measurement1), partial(show, stage=Stages.Measurement2)

    plot_sync, plot_cal, plot_exp1, plot_exp2 = \
        partial(plot, stage=Stages.TimeSynchro), \
        partial(plot, stage=Stages.Calibration), \
        partial(plot, stage=Stages.Measurement1), partial(plot, stage=Stages.Measurement2)

    filter_convolve, filter_fft, filter_savgol, filter_uniform, filter_lowess, filter_gauss, filter_spline = \
        partial(apply_filter, name='convolve'), partial(apply_filter, name='fft'), partial(apply_filter, name='savgol'), \
        partial(apply_filter, name='uniform'), partial(apply_filter, name='lowess'), partial(apply_filter, name='gaussian'), \
        partial(apply_filter, name='spline')

    set_eps_exp1 = set_eps_exp

    root.connect_callbacks(globals())

    root._app.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
