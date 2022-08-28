import numpy as np
import math
import dataclasses

import wall as wl
import constant as ct


@dataclasses.dataclass
class BuildingPart:

    # 部位名称
    name: str

    # 面積
    area: float

    # 温度差係数
    temp_diff_coeff: float

    # 室内側表面熱伝達率[W/(m2･K)]
    hi: float

    # 熱貫流率[W/(m2･K)]
    u_value: float

    # 放射空調の発熱比率
    flr: float

    # 応答係数の初項
    rfa0: float
    rft0: float

    # 公比
    r: float

    # 指数項別応答係数
    rfa1: np.ndarray
    rft1: np.ndarray

# end::tagname[]


class Room:

    def __init__(self, air_volume: float,
                 vent_volume: float,
                 beta: float):
        """
        Roomクラスの初期化
        :param air_volume: 室気積[m3]
        :param vent_volume: 室換気量[m3/s]
        :param beta: 放射空調の即時対流成分比率[－]（0～1）
        """

        # 室気積[m3]
        self.air_volume = air_volume

        # 室の熱容量[J/K] 空気+家財
        self.cap = self.air_volume * ct.rho_a * ct.c_a + self.air_volume * ct.c_frt * 1000.0

        # 換気量[m3/h] →[m3/s]
        self.vent_volume = vent_volume

        # 放射暖房の対流比率β
        self.beta = beta

        # 部位情報
        self.building_parts = []

    def building_part_append(self,
                             name: str,
                             area: float,
                             temp_diff_coeff: float,
                             hi: float,
                             flr: float,
                             u_value: float,
                             rfa0: float,
                             rft0: float,
                             r: np.ndarray,
                             rfa1: np.ndarray,
                             rft1: np.ndarray):
        """
        建物の部位を追加登録する関数
        :param name: 部位名称
        :param area: 部位面積[m2]
        :param temp_diff_coeff: 隣室温度差係数（0～1）
        :param hi: 室内側総合熱伝達率[W/(m2･K)]
        :param flr: 放射空調の当該部位の吸収比率[－]（0～1）
        :param u_value: 部位の熱貫流率[W/(m2･K)]
        :param rfa0: 部位の吸熱応答の初項[m2･K/W]
        :param rft0: 部位の貫流応答の初項[－]
        :param r: 公比
        :param rfa1: 部位の指数項別吸熱応答係数[m2･K/W]
        :param rft1: 部位の指数項別貫流応答係数[－]
        :return:
        """

        self.building_parts.append(BuildingPart(
            name=name,
            area=area,
            temp_diff_coeff=temp_diff_coeff,
            hi=hi,
            u_value=u_value,
            flr=flr,
            rfa0=rfa0,
            rft0=rft0,
            r=r,
            rfa1=rfa1,
            rft1=rft1
        ))

    def calc_f(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        係数fの計算（初期に一度だけ計算すれば求まる係数）
        :return:
            係数 f_wsr
            係数 f_wqr
            係数 f_wsrs
            係数 f_fscs
            係数 f_wqss
        """

        # 定常状態スタート時の畳み込み積分
        convolution_a_js = np.array([np.sum(bp.rfa1 / (1.0 - bp.r)) for bp in self.building_parts])
        convolution_t_js = np.array([np.sum(bp.rft1 / (1.0 - bp.r)) for bp in self.building_parts])
        temp_diff_coeff_js = np.array([bp.temp_diff_coeff for bp in self.building_parts])
        u_value_js = np.array([bp.u_value for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])

        f_ar_js = temp_diff_coeff_js * u_value_js * convolution_a_js
        f_ao_js = - f_ar_js
        f_aq_js = u_value_js / hi_js * convolution_a_js
        f_tr_js = (1.0 - temp_diff_coeff_js) * convolution_t_js
        f_to_js = temp_diff_coeff_js * convolution_t_js

        rfa0_js = np.array([bp.rfa0 for bp in self.building_parts])
        rft0_js = np.array([bp.rft0 for bp in self.building_parts])
        flr_js = np.array([bp.flr for bp in self.building_parts])
        area_js = np.array([bp.area for bp in self.building_parts])
        temp_js = 1.0 + rfa0_js * hi_js
        f_wsr_js = (rfa0_js * hi_js + (1.0 - temp_diff_coeff_js) * rft0_js) / temp_js
        f_wqr_js = rfa0_js / temp_js * flr_js / area_js * (1.0 - self.beta)
        f_wsrs_js = (f_ar_js + f_tr_js) / temp_js
        f_wscs_js = (rft0_js * temp_diff_coeff_js + f_ao_js + f_to_js) / temp_js
        f_wqss_js = (rfa0_js + f_aq_js) / temp_js

        return (f_wsr_js, f_wqr_js, f_wsrs_js, f_wscs_js, f_wqss_js)

    def calc_a0(self, f_wsr_js: np.ndarray) -> float:
        """
        係数a0の計算（初期に一度だけ計算すれば求まる係数）
        :param f_wsr_js: 係数 f_wsr_js
        :return:
            係数 a0
        """

        # 各種numpy配列を作成
        a_js = np.array([bp.area for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])

        a0 = np.sum(a_js * hi_js * (1.0 - f_wsr_js)) + ct.c_a * ct.rho_a * self.vent_volume

        return a0

    def calc_eps(self, a0: float) -> float:
        """
        係数εの計算（初期に一度だけ計算すれば求まる係数）
        :param a0: 係数 a0
        :return:
        """

        eps = np.exp(- a0 / self.cap * ct.preheat_time)

        return eps

    def calc_b(self, f_wsrs_js: np.ndarray, f_wqr_js: np.ndarray) -> (float, float):
        """
        係数b0、b1の計算（初期に一度だけ計算すれば求まる係数）
        :param f_wsrs_js: 係数 f_wsrs_js
        :param f_wqr_js: 係数 f_wqr_js
        :return:
        """

        # 各種numpy配列を作成
        a_js = np.array([bp.area for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])
        flr_js = np.array([bp.flr for bp in self.building_parts])
        rfa0_js = np.array([bp.rfa0 for bp in self.building_parts])

        b0 = np.sum(a_js * hi_js * f_wsrs_js)

        b1 = self.beta + np.sum(a_js * hi_js * f_wqr_js)

        return (b0, b1)

    def calc_b2(
        self,
        f_wqss_js: np.ndarray,
        f_wscs_js: np.ndarray,
        theta_o_s: float,
        theta_eo_s_js: float,
        q_sol_s_js: float,
        H_n: float) -> float:
        """
        係数b2の計算（毎時計算が必要）
        :param f_wqss_js: 係数 f_wqss_js
        :param f_wscs_js: 係数 f_wscs_js
        :param theta_o_s: 外気温度[℃]
        :param theta_eo_s_js: 部位j裏面の相当外気温度[℃]
        :param q_sol_s_js: 部位jの室内表面の透過日射熱取得[W/m2]
        :param H_n: 内部発熱[W]
        :return:
        """

        # 各種numpy配列を作成
        n_bndrs = len(self.building_parts)
        a_js_ns = np.tile(np.array([bp.area for bp in self.building_parts]).reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        hi_js_ns = np.tile(np.array([bp.hi for bp in self.building_parts]).reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        f_wqss_js_ns = np.tile(f_wqss_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        f_wscs_js_ns = np.tile(f_wscs_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))

        b2 = np.sum(a_js_ns * hi_js_ns * (f_wqss_js_ns * q_sol_s_js + f_wscs_js_ns * theta_eo_s_js), axis=0)\
                  + ct.c_a * ct.rho_a * self.vent_volume * theta_o_s + H_n

        return b2

    def calc_c(
        self, k_c: float,
        k_r: float,
        f_ot_js: np.ndarray,
        f_wsr_js: np.ndarray,
        f_wsrs_js: np.ndarray,
        f_wqr_js: np.ndarray) \
        -> (float, float, float):
        """
        係数cの計算（初期に一度だけ計算すれば求まる係数）
        :param k_c: 人体表面の対流熱伝達比率
        :param k_r: 人体表面の放射熱伝達比率
        :param f_ot_js: 係数 f_ot_js
        :param f_wsr_js: 係数 f_wsr_js
        :param f_wsrs_js: 係数 f_wsrs_js
        :param f_wqr_js: 係数 f_wqr_js
        :return:
        """

        temp = k_c + k_r * np.sum(f_ot_js * f_wsr_js)

        c0 = 1.0 / temp

        c1 = - k_r * np.sum(f_ot_js * f_wsrs_js) / temp

        c2 = - k_r * np.sum(f_ot_js * f_wqr_js) / temp

        return (c0, c1, c2)

    def calc_c3(
        self,
        f_ot_js: np.ndarray,
        f_wsr_js: np.ndarray,
        f_wscs_js: np.ndarray,
        f_wqss_js: np.ndarray,
        k_c: float,
        k_r: float,
        theta_eo_s_js: float,
        q_sol_s_js: float
        ) -> float:
        """
        係数c3の計算（毎時計算が必要）
        :param f_ot_js: 係数 f_ot_js
        :param f_wsr_js: 係数 f_wsr_js
        :param f_wscs_js: 係数 f_wscs_js
        :param f_wqss_js: 係数 f_wqss_js
        :param k_c: 人体表面の対流熱伝達比率
        :param k_r: 人体表面の放射熱伝達比率
        :param theta_eo_s_js: 部位j裏面の相当外気温度[℃]
        :param q_sol_s_js: 部位jの室内表面の透過日射熱取得[W/m2]
        :return:
        """

        n_bndrs = len(f_ot_js)
        f_ot_js_ns = np.tile(f_ot_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        f_wsr_js_ns = np.tile(f_wsr_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        f_wscs_js_ns = np.tile(f_wscs_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))
        f_wqss_js_ns = np.tile(f_wqss_js.reshape(n_bndrs, 1), (1, 8760 * 4 + 1))

        temp = k_c + k_r * np.sum(f_ot_js_ns * f_wsr_js_ns)

        c3 = - k_r * np.sum(f_ot_js_ns * (f_wscs_js_ns * theta_eo_s_js + f_wqss_js_ns * q_sol_s_js)) / temp

        return c3

    def calc_d(
        self,
        eps: float,
        a0: float,
        b0: float,
        b1: float,
        c0: float,
        c1: float,
        c2: float) -> (float, float, float):
        """
        係数dの計算（初期に一度だけ計算すれば求まる係数）
        :param eps: 係数 eps
        :param a0: 係数 a0
        :param b0: 係数 b0
        :param b1: 係数 b1
        :param c0: 係数 c0
        :param c1: 係数 c1
        :param c2: 係数 c2
        :return:
        """

        temp = ((1.0 - eps) * b0 / a0 + eps - c1)
        d0 = c0 / temp

        d1 = - (1.0 - eps) / a0 / temp

        d2 = (c2 - (1.0 - eps) * b1 / a0) / temp

        return (d0, d1, d2)

    def calc_d3(
        self,
        eps: float,
        a0: float,
        b0: float,
        b2: float,
        c1: float,
        c3: float
        ) -> float:
        """
        係数d3の計算（毎時計算が必要）
        :param eps: 係数 eps
        :param a0: 係数 a0
        :param b0: 係数 b0
        :param b2: 係数 b2
        :param c1: 係数 c1
        :param c3: 係数 c3
        :return:
        """

        temp = ((1.0 - eps) * b0 / a0 + eps - c1)

        d3 = (c3 - (1.0 - eps) * b2 / a0) / temp

        return d3

    def calc_theta_rs(
        self,
        d0: float,
        d1: float,
        d2: float,
        d3: float,
        theta_ot_set: float,
        l_c: float,
        l_r: float
        ) -> float:
        """
        最低保障温度の計算（毎時計算が必要）
        :param d0: 係数 d0
        :param d1: 係数 d1
        :param d2: 係数 d2
        :param d3: 係数 d3
        :param theta_ot_set: 設定作用温度[℃]
        :param l_c: 対流空調の能力[W]（冷房能力は負）
        :param l_r: 放射空調の能力[W]（冷房能力は負）
        :return:
        """

        return d0 * theta_ot_set + d1 * l_c + d2 * l_r + d3
