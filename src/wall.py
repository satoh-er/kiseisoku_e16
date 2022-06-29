import numpy as np
from dataclasses import dataclass


class Wall:

    def __init__(self, name: float, parameter_at: float, parameter_aa: float, aa0: float, at0: float, eta: float):

        # 名称
        self.name = name

        # 応答係数のパラメータ
        self.parameter_at = parameter_at
        self.parameter_aa = parameter_aa

        # 応答係数の定常項
        self.aa0 = aa0
        self.at0 = at0

        # 日射熱取得率
        self.eta = eta


    def calc_rf(self):

        # 根の設定
        self.alpha = np.logspace(np.log10(0.000002), np.log10(0.03), 8)

        # 公比の計算
        self.r = np.exp(- self.alpha * 3600.0)

        # 応答係数の初項の計算
        self.rfa0 = self.aa0 + np.sum(self.parameter_aa / (self.alpha * 3600.0) * (1.0 - self.r) )
        self.rft0 = self.at0 + np.sum(self.parameter_at / (self.alpha * 3600.0) * (1.0 - self.r) )

        # 指数項別応答係数の計算
        self.rfa1 = self.parameter_aa / (self.alpha * 3600.0) * (1.0 - self.r) ** 2.0
        self.rft1 = self.parameter_at / (self.alpha * 3600.0) * (1.0 - self.r) ** 2.0
