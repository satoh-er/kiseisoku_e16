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

    # 方位角[rad]
    wa: float

    # 傾斜角[rad]
    wb: float

    # 温度差係数
    temp_diff_coeff: float

    # 室内側表面熱伝達率[W/(m2･K)]
    hi: float

    # 熱貫流率[W/(m2･K)]
    u_value: float

    # 日射熱取得率[－]（透明な開口部のみ指定）
    eta_value: float

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

    # 傾斜面の特性値
    wz: float = 0.0
    ww: float = 0.0
    ws: float = 0.0

    # 入射角の方向余弦
    cos_theta: np.ndarray = None

    # 入射日射量（直達と天空）
    i_d: np.ndarray = None
    i_s: np.ndarray = None

    def calc_wz_ww_ws(self):
        """
        傾斜面の特性値計算
        :return:
        """

        self.wz = math.cos(self.wb)
        self.ww = math.sin(self.wb) * math.sin(self.wa)
        self.ws = math.sin(self.wb) * math.cos(self.wa)

    def calc_cos_theta(self, sh: np.ndarray, sw: np.ndarray, ss: np.ndarray):
        """
        入射角方向余弦の計算
        :param sh: sin(h)
        :param sw: cos(h) * sin(a)
        :param ss: cos(h) * cos(a)
        :return:
        """

        self.cos_theta = np.maximum(sh * self.wz + sw * self.ww + ss * self.ws, 0.0)

    def calc_slope_sol(self, idn: np.ndarray, isky: np.ndarray):
        """
        傾斜面日射量の計算
        :param idn: 法線面直達日射量[W/m2]
        :param isky: 水平面天空日射量[W/m2]
        :return:
        """

        # 直達日射量
        self.i_d = self.cos_theta * idn

        # 天空日射量
        self.i_s = (1.0 + math.cos(self.wb)) / 2.0 * isky

    def calc_q_gt(self) -> np.ndarray:
        """
        透過日射熱取得の計算[W]
        :return: 透過日射熱取得[W]
        """

        return (self.i_d + self.i_s) * self.eta_value

# end::tagname[]


class Room:

    def __init__(self, air_volume: float,
                 vent_volume: float,
                 beta: float,
                 is_cooling_convenction: bool,
                 is_heating_convvection: bool,
                 cooling_capacity: float,
                 heating_capacity: float):
        """
        Roomクラスの初期化
        :param air_volume: 室気積[m3]
        :param vent_volume: 室換気量[m3/h]
        :param beta: 放射空調の即時対流成分比率[－]（0～1）
        :param is_cooling_convenction: 冷房が対流式か否か
        :param is_heating_convvection: 暖房が対流式か否か
        :param cooling_capacity: 冷房最大能力[kW]
        :param heating_capacity: 暖房最大能力[kW]
        """

        # 室気積[m3]
        self.air_volume = air_volume

        # 室の熱容量[J/K] 空気+家財
        self.cap = self.air_volume * ct.rho_a * ct.c_a + self.air_volume * ct.c_frt * 1000.0

        # 換気量[m3/h] →[m3/s]
        self.vent_volume = vent_volume / 3600.0

        # 放射暖房の対流比率β
        self.beta = beta

        # 部位情報
        self.building_parts = []

        # 冷房方式
        self.is_cooling_convection = is_cooling_convenction

        # 暖房方式
        self.is_heating_convection = is_heating_convvection

        # 冷房能力[kW]
        self.l_c_c = 0.0
        self.l_r_c = 0.0
        if self.is_cooling_convection:
            self.l_c_c = - cooling_capacity
        else:
            self.l_r_c = - cooling_capacity

        # 暖房能力
        self.l_c_h = 0.0
        self.l_r_h = 0.0
        if self.is_heating_convection:
            self.l_c_h = heating_capacity
        else:
            self.l_r_h = heating_capacity

        self.f_wsr_js: np.ndarray = None
        self.f_wqr_js: np.ndarray = None
        self.f_wsrs_js: np.ndarray = None
        self.f_wscs_js: np.ndarray = None
        self.f_wqss_js: np.ndarray = None
        self.f_ot_js: np.ndarray = None
        self.a0: np.ndarray = None
        self.eps: np.ndarray = None
        self.b0: np.ndarray = None
        self.b1: np.ndarray = None
        self.b2: np.ndarray = None
        self.c0: np.ndarray = None
        self.c1: np.ndarray = None
        self.c2: np.ndarray = None
        self.c3: np.ndarray = None
        self.d0: np.ndarray = None
        self.d1: np.ndarray = None
        self.d2: np.ndarray = None
        self.d3: np.ndarray = None

    def building_part_append(self,
                             name: str,
                             area: float,
                             wa: float,
                             wb: float,
                             bp_type: str,
                             temp_diff_coeff: float,
                             hi: float,
                             flr: float,
                             bp_list: list,
                             walls: wl.Wall):
        """
        建物の部位を追加登録する関数
        :param name: 部位名称
        :param area: 部位面積[m2]
        :param wa: 部位の方位角[゜]
        :param wb: 部位の傾斜角[゜]（鉛直面：90、水平面：0）
        :param bp_type: 壁体名称
        :param temp_diff_coeff: 隣室温度差係数（0～1）
        :param hi: 室内側総合熱伝達率[W/(m2･K)]
        :param flr: 放射空調の当該部位の吸収比率[－]（0～1）
        :param bp_list: 壁体名称リスト
        :param walls: 壁体クラス
        :return:
        """

        wall = walls[bp_list.index(bp_type)]
        self.building_parts.append(BuildingPart(
            name=name,
            area=area,
            wa=math.radians(wa),
            wb=math.radians(wb),
            temp_diff_coeff=temp_diff_coeff,
            hi=hi,
            u_value=wall.u_value,
            eta_value=wall.eta_value,
            flr=flr,
            rfa0=wall.rfa0,
            rft0=wall.rft0,
            r=wall.r,
            rfa1=wall.rfa1,
            rft1=wall.rft1
        ))

        # 追加した部位の傾斜面特性値の計算
        self.building_parts[-1].calc_wz_ww_ws()

    def calc_f(self):
        """
        係数fの計算（初期に一度だけ計算すれば求まる係数）
        :return:
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
        self.f_wsr_js = (rfa0_js * hi_js + (1.0 - temp_diff_coeff_js) * rft0_js) / temp_js
        self.f_wqr_js = rfa0_js / temp_js * flr_js / area_js * (1.0 - self.beta)
        self.f_wsrs_js = (f_ar_js + f_tr_js) / temp_js
        self.f_wscs_js = (rft0_js * temp_diff_coeff_js + f_ao_js + f_to_js) / temp_js
        self.f_wqss_js = (rfa0_js + f_aq_js) / temp_js

    def calc_fot_js(self):
        """
        人体に対する各部位の形態係数の計算
        ここでは、面積案分にしている
        :return:
        """

        a_js = np.array([bp.area for bp in self.building_parts])

        self.f_ot_js = a_js / np.sum(a_js)

    def calc_a0(self):
        """
        係数a0の計算（初期に一度だけ計算すれば求まる係数）
        :return:
        """

        # 各種numpy配列を作成
        a_js = np.array([bp.area for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])

        self.a0 = np.sum(a_js * hi_js * (1.0 - self.f_wsr_js)) + ct.c_a * ct.rho_a * self.vent_volume

    def calc_eps(self):
        """
        係数εの計算（初期に一度だけ計算すれば求まる係数）
        :return:
        """

        self.eps = math.exp(- self.a0 / self.cap * ct.preheat_time)

    def calc_b(self):
        """
        係数b0、b1の計算（初期に一度だけ計算すれば求まる係数）
        :return:
        """

        # 各種numpy配列を作成
        a_js = np.array([bp.area for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])
        flr_js = np.array([bp.flr for bp in self.building_parts])
        rfa0_js = np.array([bp.rfa0 for bp in self.building_parts])

        self.b0 = np.sum(a_js * hi_js * self.f_wsrs_js)

        self.b1 = self.beta + np.sum(a_js * hi_js * self.f_wqr_js)

    def calc_b2(self, theta_o_s: float, theta_eo_s_js: float, q_sol_s_js: float, H_n: float):
        """
        係数b2の計算（毎時計算が必要）
        :param theta_o_s: 外気温度[℃]
        :param theta_eo_s_js: 部位j裏面の相当外気温度[℃]
        :param q_sol_s_js: 部位jの室内表面の透過日射熱取得[W/m2]
        :param H_n: 内部発熱[W]
        :return:
        """

        # 各種numpy配列を作成
        a_js = np.array([bp.area for bp in self.building_parts])
        hi_js = np.array([bp.hi for bp in self.building_parts])

        self.b2 = np.sum(a_js * hi_js * (self.f_wqss_js * q_sol_s_js + self.f_wscs_js * theta_eo_s_js))\
                  + ct.c_a * ct.rho_a * self.vent_volume * theta_o_s + H_n

    def calc_c(self, k_c: float, k_r: float):
        """
        係数cの計算（初期に一度だけ計算すれば求まる係数）
        :param k_c: 人体表面の対流熱伝達比率
        :param k_r: 人体表面の放射熱伝達比率
        :return:
        """

        temp = k_c + k_r * np.sum(self.f_ot_js * self.f_wsr_js)

        self.c0 = 1.0 / temp

        self.c1 = - k_r * np.sum(self.f_ot_js * self.f_wsrs_js) / temp

        self.c2 = - k_r * np.sum(self.f_ot_js * self.f_wqr_js) / temp

    def calc_c3(self, k_c: float, k_r: float, theta_eo_s_js: float, q_sol_s_js: float):
        """
        係数c3の計算（毎時計算が必要）
        :param k_c: 人体表面の対流熱伝達比率
        :param k_r: 仁田表面の放射熱伝達比率
        :param theta_eo_s_js: 部位j裏面の相当外気温度[℃]
        :param q_sol_s_js: 部位jの室内表面の透過日射熱取得[W/m2]
        :return:
        """

        temp = k_c + k_r * np.sum(self.f_ot_js * self.f_wsr_js)

        self.c3 = - k_r * np.sum(self.f_ot_js * (self.f_wscs_js * theta_eo_s_js + self.f_wqss_js * q_sol_s_js)) / temp

    def calc_d(self):
        """
        係数dの計算（初期に一度だけ計算すれば求まる係数）
        :return:
        """

        temp = ((1.0 - self.eps) * self.b0 / self.a0 + self.eps - self.c1)
        self.d0 = self.c0 / temp

        self.d1 = - (1.0 - self.eps) / self.a0 / temp

        self.d2 = (self.c2 - (1.0 - self.eps) * self.b1 / self.a0) / temp

    def calc_d3(self):
        """
        係数d3の計算（毎時計算が必要）
        :return:
        """

        temp = ((1.0 - self.eps) * self.b0 / self.a0 + self.eps - self.c1)

        self.d3 = (self.c3 - (1.0 - self.eps) * self.b2 / self.a0) / temp

    def calc_theta_rs(self, theta_ot_set: float, mode: str):
        """
        最低保障温度の計算（毎時計算が必要）
        :param theta_ot_set: 設定作用温度[℃]
        :param mode: 空調モード（冷房or暖房）
        :return:
        """

        if mode == 'C':
            l_c = self.l_c_c
            l_r = self.l_r_c
        else:
            l_c = self.l_c_h
            l_r = self.l_r_h

        return self.d0 * theta_ot_set + self.d1 * l_c + self.d2 * l_r + self.d3
