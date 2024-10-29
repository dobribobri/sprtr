
# номер канала <---> длина волны
configuration = [
    [1, 0.45],
    [2, 0.6],
    [3, 0.8],
    [4, 1.0],
    [5, 1.2],
    [6, 1.38],
    [7, 1.61],
    [8, 1.8],
    [9, 2.0],
    [10, 2.15],
]

# # предполагаемое количество каналов при чтении XML-файлов
# xml_n_channels = 4

# эталонные уровни сигнала с АЦП при измерении АЧТ с заданной температурой
bb_adc_levels = {
      2.15: {
            1500: 820,
            1773: 1640,
            2000: 2530,
            2500: 5130,
            4000: 16000,
      },
      2.00:  {
            1500: 661,
            1773: 1455,
            2000: 2280,
            2500: 4793,
            4000: 16000,
      },
      1.80:  {
            1500: 500,
            1773: 1160,
            2000: 1910,
            2500: 4340,
            4000: 16000,
      },
      1.61:  {
            1500: 350,
            1773: 890,
            2000: 1546,
            2500: 3844,
            4000: 16000,
      },
      1.38:  {
            1500: 195,
            1773: 573,
            2000: 1093,
            2500: 3137,
            4000: 16000,
      },
      1.20: {
            1500: 104,
            1773: 368,
            2000: 760,
            2500: 2530,
            4000: 16000,
      },
      1.00: {
            1500: 39,
            1773: 170,
            2000: 432,
            2500: 1803,
            4000: 16000,
      },
      0.80:  {
            1500: 9,
            1773: 59,
            2000: 179,
            2500: 1058,
            4000: 16000,
      },
      0.60:  {
            1500: 1,
            1773: 9,
            2000: 39,
            2500: 444,
            4000: 16000,
      },
      0.45: {
            1500: 1e-3,
            1773: 1,
            2000: 5,
            2500: 135,
            4000: 16000,
      },
}
