from re import A
import numpy as np


def moving_average_ns(data: np.ndarray) -> np.ndarray:
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
    return ma_data[2:-1]

def moving_average_js_ns(data: np.ndarray) -> np.ndarray:
    """Numpy配列を移動平均する（2次元配列）

    Args:
        data (np.ndarray): 移動平均するNumpy配列

    Returns:
        np.ndarray: 移動平均後のNumpy配列
    """

    # 最後の3要素を最初に挿入
    data_copy = np.insert(data, [0], data[:,-3:], axis=1)

    # 移動平均の個数
    num = 4

    # 平均化する重みのリスト
    b = np.ones(num,) / num

    ma_data = np.zeros((data_copy.shape))
    ma_data[:,:] = np.nan
    # 移動平均の実施
    for i in range(data_copy.shape[0]):
        ma_data[i,num-1:] = np.convolve(data_copy[i,:], b, 'valid')
        ma_data = ma_data

    # 余分な要素を削除
    return ma_data[:, 3: ]

if __name__ == '__main__':

    data = np.arange(14).reshape(2, 7)
    mv_data_2d = moving_average_js_ns(data)
    print(data)
    
    print(mv_data_2d)
    mv_data_2d_dsh = mv_data_2d.flatten().reshape([2, 7])

    data = np.arange(7)
    mv_data = moving_average_ns(data)
    print(data)
    print(mv_data)
