import numpy as np


def moving_average(data: np.ndarray) -> np.ndarray:
    """Numpy配列を移動平均する

    Args:
        data (np.ndarray): 移動平均するNumpy配列

    Returns:
        np.ndarray: 移動平均後のNumpy配列
    """

    # 最後の3要素を最初に挿入
    data_copy = np.insert(data, 0, data[-3:])

    # 移動平均の個数
    num = 4

    # 平均化する重みのリスト
    b = np.ones(num) / num

    # 移動平均の実施
    ma_data = np.convolve(data_copy, b, 'same')
    # 余分な要素を削除
    return ma_data[2: -1]