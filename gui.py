from typing import Union, Tuple
from functools import partial

from formation import AppBuilder

from tkinter import Variable
from tkinter import Toplevel
from tkinter import filedialog, messagebox
from tkinter import Label, Spinbox, Button
from tkinter import NORMAL, DISABLED
from tkinter import TclError

import core.initials as initials
from core.session import Session, Channel, Stages

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# MENU Файл
def new_session(event=None):
    global session, session_filepath

    session = Session(channels=channels_initial)
    session_filepath = None

    update_interface()


def open_session(event=None):
    global session, session_filepath

    filepath = filedialog.askopenfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])

    if filepath:
        session = Session.load(filepath)
        session_filepath = filepath
        update_interface()


def save_session(event=None):
    if session_filepath:
        update_session()
        session.save(session_filepath)
    else:
        save_as_session()


def save_as_session(event=None):
    global session_filepath
    update_session()
    session_filepath = filedialog.asksaveasfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])
    session.save(session_filepath)


def on_close(event=None):
    # session.save('session.json')
    print("До свидания!")
    exit(0)


# MENU Инструменты
# noinspection PyBroadException
def set_gain(event=None, T=2500):
    global session
    session.T_gain_cal = T
    try:
        session.set_gain()
    except TypeError:
        messagebox.showerror(title='Ошибка', message='Невозможно рассчитать коэффициенты усиления. \n Данные сеанса неверные либо отсутствуют')
    except Exception as e:
        messagebox.showerror(title='Ошибка', message=str(e))
    finally:
        update_interface()
        root.notebook_main.select(Stages.Calibration.value + 1)


def apply_filter(event=None, name='fft'):
    global session
    global t_start, t_stop

    session.apply_filter(filter_name=name, t_start=t_start, t_stop=t_stop)

    plot()


class FilterParametersDialog:
    def __init__(self, parent):

        self.filter_parameters = initials.filter_parameters

        top = self.top = Toplevel(parent)
        top.title('Параметры фильтров')
        top.geometry('{:.0f}x{:.0f}'.format(720, 400))

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
    inputDialog = FilterParametersDialog(root._app)
    root._app.wait_window(inputDialog.top)
    session.filter_params = inputDialog.filter_parameters


def filter_clear(event=None):
    global session
    session.remove_filters()


# Работа с фалами
def load_channel(event=None, channel=0, stage=Stages.Measurement):
    global session, root

    print('Globals :: load_channel() | channel {} | stage {}'.format(channel, stage))

    update_session()

    index = channel
    session.stage = stage

    filepath = filedialog.askopenfilename(filetypes=[('Текстовый файл', '*.txt'), ('Файл WFM', '*.wfm')])
    if not filepath:
        return

    try:
        ext = re.split(r'\.', filepath)
        ext = ext[-1]
    except TypeError:
        return

    # noinspection PyBroadException
    try:
        session.read_channel(index=index, filepath=filepath, format_=ext)

        match stage.value:
            case Stages.TimeSynchro.value:
                sync_load_buttons[index].configure(background="lightgreen")
            case Stages.Calibration.value:
                cal_load_buttons[index].configure(background="lightgreen")
            case Stages.Measurement.value:
                exp_load_buttons[index].configure(background="lightgreen")

    except Exception as e:
        messagebox.showerror(title='Ошибка чтения файла', message=str(e))
        match stage.value:
            case Stages.TimeSynchro.value:
                sync_load_buttons[index].configure(background="pink")
            case Stages.Calibration.value:
                cal_load_buttons[index].configure(background="pink")
            case Stages.Measurement.value:
                exp_load_buttons[index].configure(background="pink")


def load_all(event=None, stage=Stages.Measurement):
    global session, root

    print('Globals :: load_all()')

    update_session()

    session.stage = stage

    filepaths = filedialog.askopenfilenames(filetypes=[('Файл CSV', '*.csv'), ('Файл DAT', '*.dat'),
                                                       ('Текстовый файл', '*.txt'), ('Файл WFM', '*.wfm')])
    if not filepaths:
        return

    extensions = []
    try:
        for filepath in filepaths:
            ext = re.split(r'\.', filepath)
            extensions.append(ext[-1])
    except TypeError:
        return

    if len(np.unique(extensions)) > 1:
        messagebox.showerror(title='Ошибка', message="Пожалуйста, выберите файлы одного и того же типа")

    ext = np.unique(extensions)[0]

    # noinspection PyBroadException
    try:
        session.read_channels(filepaths=filepaths, format_=ext)
    except Exception as e:
        messagebox.showerror(title='Ошибка чтения файла', message=str(e))

    update_interface()


def clear_all(event=None, stage=Stages.Measurement):
    global session, root

    session.erase()
    update_interface()


# Визуализация
# noinspection PyUnresolvedReferences
def plt_on_button_press(event):
    global area
    global t_start
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


def show(event=None, stage=Stages.Measurement):  # показать исходные данные
    global root
    global figure, axes
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


def plot(event=None, stage=Stages.Measurement):  # показать результат обработки
    global result

    update_session()
    fig, ax = plt.subplots()
    for j, channel in enumerate(session.channels_valid):
        ax.plot(channel.time_synced, channel.data_calibrated,
                label='Канал #{}'.format(channel.number))
    ax.set_xlabel('Время')
    ax.set_ylabel('Сигнал')
    ax.legend(loc="upper right", frameon=True)
    ax.grid(ls=':', color='gray')

    if stage.value == Stages.Measurement.value:
        if result is not None:
            time, T = result

            ax = plt.twinx(ax)
            ax.plot(time, T, color='blue', lw=2, label='Температура')
            ax.legend(loc="center left", frameon=True)

    plt.tight_layout()
    plt.show()

    if root.mode_3D.get() and stage.value == Stages.Measurement.value and result is not None:
        pass


# Вкладка "Синхронизация"
def apply_sync(event=None):
    global session, root
    update_session()
    session.set_timedelta()
    update_interface()


# Вкладка "Калибровка"
def set_eps_cal(event=None):  # Излучательная способность эталона
    global session, root

    eps = 1
    filepath = filedialog.askopenfilename(filetypes=[("Таблица Excel", ".xlsx .xls")])

    if filepath:
        # noinspection PyBroadException
        try:
            df = pd.read_excel(filepath, index_col=None, header=None)
            eps = dict(df.to_numpy().tolist())
        except Exception as e:
            messagebox.showerror(title='Ошибка', message=str(e))
            eps = 1

    if root.calibration_type.get():  # Относительная калибровка
        session.eps_rel_cal = eps
    else:  # Абсолютная калибровка
        session.eps_abs_cal = eps


def apply_cal(event=None):
    global session
    if root.calibration_type.get():  # Относительная калибровка
        session.relative_calibration()
    else:
        session.absolute_calibration()
    update_interface()


# Вкладка "Эксперимент"
def set_eps_exp(event=None):
    global session, root

    eps = 1
    filepath = filedialog.askopenfilename(filetypes=[("Таблица Excel", ".xlsx .xls")])

    if filepath:
        # noinspection PyBroadException
        try:
            df = pd.read_excel(filepath, index_col=None, header=None)
            eps = dict(df.to_numpy().tolist())
        except Exception as e:
            messagebox.showerror(title='Ошибка', message=str(e))
            eps = 1

    session.eps_sample = eps


def calculate_temperatures(event=None):
    global session
    update_session()

    global t_start, t_stop
    global result

    try:
        result = session.get_temperature(t_start=t_start, t_stop=t_stop, parallel=True)
    except Exception as e:
        messagebox.showerror('Ошибка расчета температуры', str(e))


def export_temperatures(event=None):
    pass


# Основное
def update_session(*args):
    global session

    # root._app.update()
    # root._app.update_idletasks()

    session.used = list(map(lambda var: var.get(), usage))

    if root.calibration_type.get():  # Относительная калибровка
        session.T_rel_cal = root.T_cal.get()
    else:  # Абсолютная калибровка
        session.T_abs_cal = root.T_cal.get()

    session.tp_n_interp_exp = None
    if root.tp_interp_exp.get():
        session.tp_n_interp_exp = root.tp_n_interp_exp.get()

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
                    case Stages.Measurement.value:
                        mask = exp_mask[j].get()
                        if root.fr_interp_exp.get():
                            n_interp = root.fr_n_interp_exp.get()

                session.channels[j].Series[stage.value].n_interp = n_interp
                session.channels[j].Series[stage.value].mask = mask

        except TclError:
            pass

    tab_index = root.notebook_main.index(root.notebook_main.select())
    stage_val = tab_index - 1
    if 0 <= stage_val < 3:
        session.stage = Stages(stage_val)

    # update_interface()


def update_interface(*args):
    global root

    for stage in Stages:
        n_interp = session.first_channel.Series[stage.value].n_interp
        match stage.value:
            case Stages.TimeSynchro.value:
                root.fr_interp_sync.set(bool(n_interp))
                # root.checkbutton_fr_interp_sync.update()
                root.fr_n_interp_sync.set(n_interp)
                # root.entry_n_fr_interp_sync.update()
            case Stages.Calibration.value:
                root.fr_interp_cal.set(bool(n_interp))
                # root.checkbutton_fr_interp_cal.update()
                root.fr_n_interp_cal.set(n_interp)
                # root.entry_n_fr_interp_cal.update()
            case Stages.Measurement.value:
                root.fr_interp_exp.set(bool(n_interp))
                # root.checkbutton_fr_interp_exp.update()
                root.fr_n_interp_exp.set(n_interp)
                # root.entry_n_fr_interp_exp.update()

    for j in range(10):

        if j in session.used_indexes:
            usage_checkbuttons[j].select()
        else:
            usage_checkbuttons[j].deselect()
        # usage_checkbuttons[j].update()

        lambdas[j].set(session.channels[j].wavelength)
        # lambdas_spinboxes[j].update()

        gains[j].set(session.channels[j].gain)
        # gains_spinboxes[j].update()

        deltas[j].set(session.channels[j].timedelta)
        # deltas_spinboxes[j].update()

        alphas[j].set(session.channels[j].alpha)
        # alphas_spinboxes[j].update()

        for stage in Stages:
            mask = session.channels[j].Series[stage.value].mask
            loaded = isinstance(session.channels[j].Series[stage.value].data, np.ndarray)
            match stage.value:
                case Stages.TimeSynchro.value:
                    sync_mask[j].set(mask)
                    # sync_checkbuttons[j].update()
                    sync_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)
                case Stages.Calibration.value:
                    cal_mask[j].set(mask)
                    # cal_checkbuttons[j].update()
                    cal_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)
                case Stages.Measurement.value:
                    exp_mask[j].set(mask)
                    # exp_checkbuttons[j].update()
                    exp_load_buttons[j].configure(background="lightgreen" if loaded else original_button_color)

    root.tp_interp_exp.set(bool(session.tp_n_interp_exp))
    # root.checkbutton_tp_interp_calc_exp.update()
    root.tp_n_interp_exp.set(session.tp_n_interp_exp)
    # root.entry_n_tp_interp_calc_exp.update()

    # root._app.update()
    # root._app.update_idletasks()

    current_stage = session.stage
    root.notebook_main.select(current_stage.value + 1)


def update_usage(*args):
    global session, root

    for j in range(10):
        if usage[j].get():
            lambdas_spinboxes[j].configure(state=NORMAL)
            deltas_spinboxes[j].configure(state=NORMAL)
            alphas_spinboxes[j].configure(state=NORMAL)
            gains_spinboxes[j].configure(state=NORMAL)

            sync_checkbuttons[j].configure(state=NORMAL)
            cal_checkbuttons[j].configure(state=NORMAL)
            exp_checkbuttons[j].configure(state=NORMAL)

            sync_load_buttons[j].configure(state=NORMAL)
            cal_load_buttons[j].configure(state=NORMAL)
            exp_load_buttons[j].configure(state=NORMAL)

        else:
            for stage in Stages:
                session.channels[j].Series[stage.value].mask = False
                match stage.value:
                    case Stages.TimeSynchro.value:
                        sync_mask[j].set(False)
                        # sync_checkbuttons[j].update()
                    case Stages.Calibration.value:
                        cal_mask[j].set(False)
                        # cal_checkbuttons[j].update()
                    case Stages.Measurement.value:
                        exp_mask[j].set(False)
                        # exp_checkbuttons[j].update()

            lambdas_spinboxes[j].configure(state=DISABLED)
            deltas_spinboxes[j].configure(state=DISABLED)
            alphas_spinboxes[j].configure(state=DISABLED)
            gains_spinboxes[j].configure(state=DISABLED)

            sync_checkbuttons[j].configure(state=DISABLED)
            cal_checkbuttons[j].configure(state=DISABLED)
            exp_checkbuttons[j].configure(state=DISABLED)

            sync_load_buttons[j].configure(state=DISABLED)
            cal_load_buttons[j].configure(state=DISABLED)
            exp_load_buttons[j].configure(state=DISABLED)


if __name__ == "__main__":
    root = AppBuilder(path="window.xml")

    # Сессия
    channels_initial = [Channel(wavelength=wavelength) for _, wavelength in initials.configuration]
    # session: Session = Session(channels=channels_initial)
    session: Session = Session.load('session.json')
    session_filepath = None

    figure, axes, area = None, None, None
    t_start, t_stop = None, None  # временной интервал
    result: Union[Tuple[np.ndarray, np.ndarray], None] = None

    root.notebook_main.select(0)

    # Checkbuttons
    usage_checkbuttons, sync_checkbuttons, cal_checkbuttons, exp_checkbuttons = [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("usage_checkbuttons.append(root.checkbutton_channel_{})".format(chn))
        exec("sync_checkbuttons.append(root.checkbutton_ch{}_sync)".format(chn))
        exec("cal_checkbuttons.append(root.checkbutton_ch{}_cal)".format(chn))
        exec("exp_checkbuttons.append(root.checkbutton_ch{}_calc_exp)".format(chn))

    # Spinboxes
    lambdas_spinboxes, deltas_spinboxes, alphas_spinboxes, gains_spinboxes = \
        [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("lambdas_spinboxes.append(root.spinbox_lambda_{})".format(chn))
        exec("deltas_spinboxes.append(root.spinbox_sync_{})".format(chn))
        exec("alphas_spinboxes.append(root.spinbox_alpha_{})".format(chn))
        exec("gains_spinboxes.append(root.spinbox_gain_{})".format(chn))

    # Load buttons
    sync_load_buttons, cal_load_buttons, exp_load_buttons = [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("sync_load_buttons.append(root.button_load_ch{}_sync)".format(chn))
        exec("cal_load_buttons.append(root.button_load_ch{}_cal)".format(chn))
        exec("exp_load_buttons.append(root.button_load_ch{}_exp)".format(chn))

    original_button_color = root.button_load_all_exp.cget("background")

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
    sync_mask, cal_mask, exp_mask = [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("sync_mask.append(root.sync_ch{})".format(chn))  # Синхронизация
        exec("cal_mask.append(root.cal_ch{})".format(chn))    # Калибровка
        exec("exp_mask.append(root.temp_ch{})".format(chn))   # Эксперимент

    update_interface()

    set_gain_T1500_depr, set_gain_T1773_depr, set_gain_T2000_depr, set_gain_T2500_depr = \
        partial(set_gain, T=1500), partial(set_gain, T=1773), partial(set_gain, T=2000), partial(set_gain, T=2500)

    s = ''
    for i in range(10):
        chn = str(i + 1).zfill(2)
        s += 'load_ch{}_sync, load_ch{}_cal, load_ch{}_exp, '.format(chn, chn, chn)
    s = s[:-2] + ' = '
    for i in range(10):
        s += 'partial(load_channel, channel={}, stage=Stages.TimeSynchro), '.format(i)
        s += 'partial(load_channel, channel={}, stage=Stages.Calibration), '.format(i)
        s += 'partial(load_channel, channel={}, stage=Stages.Measurement), '.format(i)
    exec(s[:-2])

    load_all_sync, load_all_cal, load_all_exp = partial(load_all, stage=Stages.TimeSynchro), \
        partial(load_all, stage=Stages.Calibration), partial(load_all, stage=Stages.Measurement)
    clear_all_sync, clear_all_cal, clear_all_exp = partial(clear_all, stage=Stages.TimeSynchro), \
        partial(clear_all, stage=Stages.Calibration), partial(clear_all, stage=Stages.Measurement)

    show_sync, show_cal, show_exp = partial(show, stage=Stages.TimeSynchro), partial(show, stage=Stages.Calibration), \
        partial(show, stage=Stages.Measurement)

    plot_sync, plot_cal, plot_exp = partial(plot, stage=Stages.TimeSynchro), partial(plot, stage=Stages.Calibration), \
        partial(plot, stage=Stages.Measurement)

    filter_convolve, filter_fft, filter_savgol, filter_uniform, filter_lowess, filter_gauss, filter_spline = \
        partial(apply_filter, name='convolve'), partial(apply_filter, name='fft'), partial(apply_filter, name='savgol'), \
        partial(apply_filter, name='uniform'), partial(apply_filter, name='lowess'), partial(apply_filter, name='gaussian'), \
        partial(apply_filter, name='spline')

    root.connect_callbacks(globals())

    root._app.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
