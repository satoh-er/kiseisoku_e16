import pandas as pd
import numpy as np

class Weather:

    def __init__(self, weather_pd: pd.DataFrame):

        # 外気温度[℃]
        self.theta_o = self.interpolate(weather_data=weather_pd['外気温[℃]'].values, rolling=True)

        # 法線面直達日射量[W/m2]
        self.idn = self.interpolate(weather_data=weather_pd['法線面直達日射量 [W/m2]'].values, rolling=True)

        # 水平面天空日射量[W/m2]
        self.isky = self.interpolate(weather_data=weather_pd['水平面天空日射量 [W/m2]'].values, rolling=True)

        # 太陽高度[゜] → [rad]
        self.h_s = self.interpolate(weather_data=np.radians(weather_pd['太陽高度角[度]'].values), rolling=True)

        # 太陽方位角[゜] → [rad]
        self.a_s = self.interpolate(weather_data=np.radians(weather_pd['太陽方位角[度]'].values), rolling=True)

        self.sh = np.sin(self.h_s)
        self.sw = np.cos(self.h_s) * np.sin(self.a_s)
        self.ss = np.cos(self.h_s) * np.cos(self.a_s)

    def interpolate(self, weather_data: np.ndarray, rolling: bool) -> np.ndarray:
        """
        1時間ごとの8760データを15分間隔のデータに補間する。
        '15m': 15分間隔の場合、 n = 8760 * 4 = 35040

        Args:
            weather_data: 1時間ごとの気象データ [8760]
            rolling: rolling するか否か。データが1時始まりの場合は最終行の 12/31 24:00 のデータを 1/1 0:00 に持ってくるため、この値は True にすること。

        Returns:
            指定する時間間隔に補間された気象データ [n]
        """

        # 補間比率の係数
        alpha = np.array([1.0, 0.75, 0.5, 0.25])

        # 補間元データ1, 補間元データ2
        if rolling:
            # 拡張アメダスのデータが1月1日の1時から始まっているため1時間ずらして0時始まりのデータに修正する。
            data1 = np.roll(weather_data, 1)     # 0時=24時のため、1回分前のデータを参照
            data2 = weather_data
        else:
            data1 = weather_data
            data2 = np.roll(weather_data, -1)

        # 直線補完 8760×4 の2次元配列
        data_interp_2d = alpha[np.newaxis, :] * data1[:, np.newaxis] + (1.0 - alpha[np.newaxis, :]) * data2[:, np.newaxis]

        # 1次元配列へ変換
        data_interp_1d = data_interp_2d.flatten()

        return data_interp_1d
