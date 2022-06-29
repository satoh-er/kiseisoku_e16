import pandas as pd
import numpy as np

class Weather:

    def __init__(self, weather_pd: pd.DataFrame):

        # 外気温度[℃]
        self.theta_o = weather_pd['外気温[℃]'].values

        # 法線面直達日射量[W/m2]
        self.idn = weather_pd['法線面直達日射量 [W/m2]'].values

        # 水平面天空日射量[W/m2]
        self.isky = weather_pd['水平面天空日射量 [W/m2]'].values

        # 太陽高度[゜] → [rad]
        self.h_s = np.radians(weather_pd['太陽高度角[度]'].values)

        # 太陽方位角[゜] → [rad]
        self.a_s = np.radians(weather_pd['太陽方位角[度]'].values)

        self.sh = np.sin(self.h_s)
        self.sw = np.cos(self.h_s) * np.sin(self.a_s)
        self.ss = np.cos(self.h_s) * np.cos(self.a_s)
