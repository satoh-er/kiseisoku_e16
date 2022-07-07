import numpy as np

import constant as ct

class Wall:

    def __init__(self, name: float, parameter_at: float, parameter_aa: float, aa0: float, at0: float, eta_value: float, u_value: float):

        # 名称
        self.name = name

        # 応答係数のパラメータ
        self.parameter_at = parameter_at
        self.parameter_aa = parameter_aa

        # 応答係数の定常項
        self.aa0 = aa0
        self.at0 = at0

        # 日射熱取得率
        self.eta_value = eta_value

        # 熱貫流率[W/(m2･K)]
        self.u_value = u_value

    def calc_rf(self):

        # 根の設定
        self.alpha = np.logspace(np.log10(0.000002), np.log10(0.03), 8)

        # 公比の計算
        self.r = np.exp(- self.alpha * ct.preheat_time)

        # 応答係数の初項の計算
        self.rfa0 = self.aa0 + np.sum(self.parameter_aa / (self.alpha * ct.preheat_time) * (1.0 - self.r) )
        self.rft0 = self.at0 + np.sum(self.parameter_at / (self.alpha * ct.preheat_time) * (1.0 - self.r) )

        # 指数項別応答係数の計算
        self.rfa1 = - self.parameter_aa / (self.alpha * ct.preheat_time) * (1.0 - self.r) ** 2.0
        self.rft1 = - self.parameter_at / (self.alpha * ct.preheat_time) * (1.0 - self.r) ** 2.0
