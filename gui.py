
from formation import AppBuilder
from tkinter import *
from core.session import Session


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


if __name__ == "__main__":
    root = AppBuilder(path="window.xml")

    session = Session()

    root.connect_callbacks(globals())
    root._app.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
