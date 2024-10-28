
from formation import AppBuilder
from tkinter import *
import core.initials as initials
# from core.session import Session


# MENU Файл
def new_session(event=None):
    pass


def open_session(event=None):
    pass


def save_session(event=None):
    pass


def save_as_session(event=None):
    pass


def on_close(event=None):
    session.save()
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
def update_state(*args):
    usage_vals = []
    lambda_vals, delta_vals, alpha_vals, gain_vals = [], [], [], []
    for j in range(10):
        usage_vals.append(usage[j].get())
        lambda_vals.append(lambdas[j].get())
        delta_vals.append(deltas[j].get())
        alpha_vals.append(alphas[j].get())
        gain_vals.append(gains[j].get())
    print(lambda_vals, delta_vals, alpha_vals, gain_vals)


if __name__ == "__main__":
    root = AppBuilder(path="window.xml")

    root.notebook_main.select(3)

    # session = Session()
    usage = [root.use_ch01, root.use_ch02, root.use_ch03, root.use_ch04, root.use_ch05,
             root.use_ch06, root.use_ch07, root.use_ch08, root.use_ch09, root.use_ch10]

    for i in range(10):
        usage[i].set(True)
        usage[i].trace('w', update_state)

    lambdas = [root.lambda01, root.lambda02, root.lambda03, root.lambda04, root.lambda05,
               root.lambda06, root.lambda07, root.lambda08, root.lambda09, root.lambda10]

    for i in range(10):
        lambdas[i].set(initials.configuration[i][1])
        lambdas[i].trace('w', update_state)

    deltas = [root.deltat01, root.deltat02, root.deltat03, root.deltat04, root.deltat05,
              root.deltat06, root.deltat07, root.deltat08, root.deltat09, root.deltat10]

    for i in range(10):
        deltas[i].set(0.0)
        deltas[i].trace('w', update_state)

    alphas = [root.alpha01, root.alpha02, root.alpha03, root.alpha04, root.alpha05,
              root.alpha06, root.alpha07, root.alpha08, root.alpha09, root.alpha10]

    for i in range(10):
        alphas[i].set(1.0)
        alphas[i].trace('w', update_state)

    gains = [root.gain01, root.gain02, root.gain03, root.gain04, root.gain05,
             root.gain06, root.gain07, root.gain08, root.gain09, root.gain10]

    for i in range(10):
        gains[i].set(1.0)
        gains[i].trace('w', update_state)

    root.connect_callbacks(globals())

    root._app.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()
