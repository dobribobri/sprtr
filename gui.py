from formation import AppBuilder
from tkinter import Toplevel, filedialog, TclError, NORMAL, DISABLED
import core.initials as initials
from core.session import Session, Channel, Stages
import os
import numpy as np


# MENU Файл
def new_session(event=None):
    global session, session_filepath

    session = Session(channels=channels_initial)
    session_filepath = None

    update_interface()


def open_session(event=None):
    global session, session_filepath
    session_filepath = filedialog.askopenfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])
    if session_filepath:
        session = Session.load(session_filepath)
    update_interface()


def save_session(event=None):
    if session_filepath:
        # update_session()
        session.save(session_filepath)
    else:
        save_as_session()


def save_as_session(event=None):
    global session_filepath
    # update_session()
    session_filepath = filedialog.asksaveasfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])
    session.save(session_filepath)


def on_close(event=None):
    # session.save('session.json')
    print("До свидания!")
    exit(0)


# MENU Инструменты
def filter_convolve(event=None):
    pass


def filter_fft(event=None):
    pass


def filter_savgol(event=None):
    pass


def filter_rect(event=None):
    pass


def filter_lowess(event=None):
    pass


def filter_gauss(event=None):
    pass


def filter_spline(event=None):
    pass


def filter_parameters(event=None):
    child_w = Toplevel(root._app)
    child_w.title("Параметры фильтров")


def filter_clear(event=None):
    pass


def set_gain_T1500_depr(event=None):
    pass


def set_gain_T1773_depr(event=None):
    pass


def set_gain_T2000_depr(event=None):
    pass


def set_gain_T2500_depr(event=None):
    pass


# Вкладка "Синхронизация"
def load_all_sync(event=None):
    pass


def clear_all_sync(event=None):
    pass


def load_ch01_sync(event=None):
    pass


def load_ch02_sync(event=None):
    pass


def load_ch03_sync(event=None):
    pass


def load_ch04_sync(event=None):
    pass


def load_ch05_sync(event=None):
    pass


def load_ch06_sync(event=None):
    pass


def load_ch07_sync(event=None):
    pass


def load_ch08_sync(event=None):
    pass


def load_ch09_sync(event=None):
    pass


def load_ch10_sync(event=None):
    pass


def show_sync(event=None):
    pass


def apply_sync(event=None):
    pass


# Вкладка "Калибровка"
def set_eps_cal(event=None):
    child_w = Toplevel(root._app)
    child_w.title("Излучательная способность эталона")


def load_all_cal(event=None):
    pass


def clear_all_cal(event=None):
    pass


def load_ch01_cal(event=None):
    pass


def load_ch02_cal(event=None):
    pass


def load_ch03_cal(event=None):
    pass


def load_ch04_cal(event=None):
    pass


def load_ch05_cal(event=None):
    pass


def load_ch06_cal(event=None):
    pass


def load_ch07_cal(event=None):
    pass


def load_ch08_cal(event=None):
    pass


def load_ch09_cal(event=None):
    pass


def load_ch10_cal(event=None):
    pass


def show_cal(event=None):
    pass


def apply_cal(event=None):
    pass


# Вкладка "Эксперимент"
def set_eps_exp(event=None):
    child_w = Toplevel(root._app)
    child_w.title("Излучательная способность образца")


def load_all_exp(event=None):
    pass


def clear_all_exp(event=None):
    pass


def load_ch01_exp(event=None):
    pass


def load_ch02_exp(event=None):
    pass


def load_ch03_exp(event=None):
    pass


def load_ch04_exp(event=None):
    pass


def load_ch05_exp(event=None):
    pass


def load_ch06_exp(event=None):
    pass


def load_ch07_exp(event=None):
    pass


def load_ch08_exp(event=None):
    pass


def load_ch09_exp(event=None):
    pass


def load_ch10_exp(event=None):
    pass


def show_exp(event=None):
    pass


def calculate_temperatures(event=None):
    pass


def show_temperatures(event=None):
    pass


def export_temperatures(event=None):
    pass


# Основное
def update_session(*args):
    global session

    # print('===== UPDATE SESSION =====')
    session.used = list(map(lambda var: var.get(), usage))

    tab_index = root.notebook_main.index(root.notebook_main.select())
    stage_val = tab_index - 1
    if 0 <= stage_val < 3:
        session.stage = Stages(stage_val)

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
                # if j + 1 in [9, 7, 5]:
                #     print('channel {} - stage {} - mask {} | {}'.format(j + 1, stage.name, mask, session.channels[j].Series[stage.value].mask))

        except TclError:
            pass

    # update_interface()


def update_interface(*args):

    global session, root
    global usage, lambdas, deltas, alphas, gains
    global sync_mask, cal_mask, exp_mask
    global usage_checkbuttons, lambdas_spinboxes, gains_spinboxes, deltas_spinboxes, alphas_spinboxes
    global sync_checkbuttons, cal_checkbuttons, exp_checkbuttons

    for stage in Stages:
        n_interp = session.first_channel.Series[stage.value].n_interp
        match stage.value:
            case Stages.TimeSynchro.value:
                root.fr_interp_sync.set(bool(n_interp))
                root.checkbutton_fr_interp_sync.update()
                root.fr_n_interp_sync.set(n_interp)
                root.entry_n_fr_interp_sync.update()
            case Stages.Calibration.value:
                root.fr_interp_cal.set(bool(n_interp))
                root.checkbutton_fr_interp_cal.update()
                root.fr_n_interp_cal.set(n_interp)
                root.entry_n_fr_interp_cal.update()
            case Stages.Measurement.value:
                root.fr_interp_exp.set(bool(n_interp))
                root.checkbutton_fr_interp_exp.update()
                root.fr_n_interp_exp.set(n_interp)
                root.entry_n_fr_interp_exp.update()

    for j in range(10):
        usage[j].set(j in session.used_indexes)
        usage_checkbuttons[j].update()

        lambdas[j].set(session.channels[j].wavelength)
        lambdas_spinboxes[j].update()

        gains[j].set(session.channels[j].gain)
        gains_spinboxes[j].update()

        deltas[j].set(session.channels[j].timedelta)
        deltas_spinboxes[j].update()

        alphas[j].set(session.channels[j].alpha)
        alphas_spinboxes[j].update()

        for stage in Stages:
            mask = session.channels[j].Series[stage.value].mask

            match stage.value:
                case Stages.TimeSynchro.value:
                    sync_mask[j].set(mask)
                    sync_checkbuttons[j].update()
                case Stages.Calibration.value:
                    cal_mask[j].set(mask)
                    cal_checkbuttons[j].update()
                case Stages.Measurement.value:
                    exp_mask[j].set(mask)
                    exp_checkbuttons[j].update()

    root.tp_interp_exp.set(bool(session.tp_n_interp_exp))
    root.checkbutton_tp_interp_calc_exp.update()
    root.tp_n_interp_exp.set(session.tp_n_interp_exp)
    root.entry_n_tp_interp_calc_exp.update()

    update_usage()

    root._app.update()
    root._app.update_idletasks()

    current_stage = session.stage
    root.notebook_main.select(current_stage.value + 1)


def update_usage(*args):
    global session

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
                        sync_checkbuttons[j].update()
                    case Stages.Calibration.value:
                        cal_mask[j].set(False)
                        cal_checkbuttons[j].update()
                    case Stages.Measurement.value:
                        exp_mask[j].set(False)
                        exp_checkbuttons[j].update()

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


# def update_lambdas(*args):
#     global session
#     try:
#         for j in range(10):
#             session.channels[j].wavelength = lambdas[j].get()
#     except TclError as e:
#         print(e)
#
#
# def update_deltas(*args):
#     global session
#     try:
#         for j in range(10):
#             session.channels[j].timedelta = deltas[j].get()
#     except TclError as e:
#         print(e)
#
#
# def update_alphas(*args):
#     global session
#     try:
#         for j in range(10):
#             session.channels[j].alpha = alphas[j].get()
#     except TclError as e:
#         print(e)
#
#
# def update_gains(*args):
#     global session
#     try:
#         for j in range(10):
#             session.channels[j].gain = gains[j].get()
#     except TclError as e:
#         print(e)
#
#
# def update_sync_mask(*args):
#     global session
#     for j in range(10):
#         session.channels[j].Series[Stages.TimeSynchro.value].mask = sync_mask[j].get()
#
#
# def update_cal_mask(*args):
#     global session
#     for j in range(10):
#         session.channels[j].Series[Stages.Calibration.value].mask = cal_mask[j].get()
#
#
# def update_exp_mask(*args):
#     global session
#     for j in range(10):
#         session.channels[j].Series[Stages.Measurement.value].mask = exp_mask[j].get()


if __name__ == "__main__":
    root = AppBuilder(path="window.xml")

    # Сессия
    channels_initial = [Channel(wavelength=wavelength) for _, wavelength in initials.configuration]
    session: Session = Session(channels=channels_initial)
    session_filepath = None

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

    # Основные переменные
    usage, lambdas, deltas, alphas, gains = [], [], [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("usage.append(root.use_ch{})".format(chn))
        exec("lambdas.append(root.lambda{})".format(chn))
        exec("deltas.append(root.deltat{})".format(chn))
        exec("alphas.append(root.alpha{})".format(chn))
        exec("gains.append(root.gain{})".format(chn))

    # for i in range(10):
    #     # # usage[i].set(True)
    #     # usage[i].trace('w', update_usage)
    #     #
    #     # # lambdas[i].set(initials.configuration[i][1])
    #     # lambdas[i].trace('w', update_lambdas)
    #     #
    #     # # deltas[i].set(0.0)
    #     # deltas[i].trace('w', update_deltas)
    #     #
    #     # # alphas[i].set(1.0)
    #     # alphas[i].trace('w', update_alphas)
    #     #
    #     # # gains[i].set(1.0)
    #     # gains[i].trace('w', update_gains)
    #     pass

    # Другие переменные
    sync_mask, cal_mask, exp_mask = [], [], []
    for i in range(10):
        chn = str(i + 1).zfill(2)
        exec("sync_mask.append(root.sync_ch{})".format(chn))  # Синхронизация
        exec("cal_mask.append(root.cal_ch{})".format(chn))    # Калибровка
        exec("exp_mask.append(root.temp_ch{})".format(chn))   # Эксперимент

    # for i in range(10):
    #     # sync_mask[i].trace('w', update_sync_mask)
    #     # cal_mask[i].trace('w', update_cal_mask)
    #     # exp_mask[i].trace('w', update_exp_mask)
    #     pass

    # update_session()
    update_interface()

    root.connect_callbacks(globals())

    root._app.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
