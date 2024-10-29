from formation import AppBuilder
from tkinter import Toplevel, filedialog, TclError
import core.initials as initials
from core.session import Session, Channel
import os
import numpy as np


# MENU Файл
def new_session(event=None):
    global session
    session = Session(channels=channels_initial)
    update_interface()


def open_session(event=None):
    global session, session_filepath
    session_filepath = filedialog.askopenfilename(filetypes=[('Параметры сессии (JSON)', '*.json')])
    if session_filepath:
        session = Session.load(session_filepath)
    update_interface()


def save_session(event=None):
    update_session()
    if session_filepath:
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


# Вкладка "Основное"
def update_session(*args):
    session.used_indexes = np.argwhere(list(map(lambda var: var.get(), usage)))
    for j in session.used_indexes:
        try:
            session.channels[j].wavelength = lambdas[j].get()
            session.channels[j].gain = gains[j].get()
            session.channels[j].timedelta = deltas[j].get()
            session.channels[j].alpha = alphas[j].get()
        except TclError:
            pass


def update_interface(*args):
    global usage, lambdas, deltas, alphas, gains

    for j in range(10):
        if j in session.used_indexes:
            exec("root.use_ch{}.set(True)".format(str(j+1).zfill(2)))
        else:
            exec("root.use_ch{}.set(False)".format(str(j+1).zfill(2)))

        lambdas[j].set(session.channels[j].wavelength)
        gains[j].set(session.channels[j].gain)
        deltas[j].set(session.channels[j].timedelta)
        alphas[j].set(session.channels[j].alpha)


if __name__ == "__main__":
    root = AppBuilder(path="window.xml")

    root.notebook_main.select(0)

    usage = [root.use_ch01, root.use_ch02, root.use_ch03, root.use_ch04, root.use_ch05,
             root.use_ch06, root.use_ch07, root.use_ch08, root.use_ch09, root.use_ch10]

    for i in range(10):
        usage[i].set(True)

    lambdas = [root.lambda01, root.lambda02, root.lambda03, root.lambda04, root.lambda05,
               root.lambda06, root.lambda07, root.lambda08, root.lambda09, root.lambda10]

    for i in range(10):
        lambdas[i].set(initials.configuration[i][1])

    deltas = [root.deltat01, root.deltat02, root.deltat03, root.deltat04, root.deltat05,
              root.deltat06, root.deltat07, root.deltat08, root.deltat09, root.deltat10]

    for i in range(10):
        deltas[i].set(0.0)

    alphas = [root.alpha01, root.alpha02, root.alpha03, root.alpha04, root.alpha05,
              root.alpha06, root.alpha07, root.alpha08, root.alpha09, root.alpha10]

    for i in range(10):
        alphas[i].set(1.0)

    gains = [root.gain01, root.gain02, root.gain03, root.gain04, root.gain05,
             root.gain06, root.gain07, root.gain08, root.gain09, root.gain10]

    for i in range(10):
        gains[i].set(1.0)

    channels_initial = [Channel(wavelength=wavelength) for wavelength in map(lambda var: var.get(), lambdas)]
    session: Session = Session(channels=channels_initial)
    session_filepath = None

    update_session()

    root.connect_callbacks(globals())

    root._app.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
