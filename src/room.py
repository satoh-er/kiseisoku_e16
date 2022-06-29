import numpy as np
import math
import dataclasses

import wall as wl


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

    # 室内側表面熱伝達率[W/(m2･K)]
    hi: float

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

        self.wz = math.cos(self.wb)
        self.ww = math.sin(self.wb) * math.sin(self.wa)
        self.ws = math.sin(self.wb) * math.cos(self.wa)

    def calc_cos_theta(self, sh: np.ndarray, sw: np.ndarray, ss: np.ndarray):

        self.cos_theta = np.maximum(sh * self.wz + sw * self.ww + ss * self.ws, 0.0)

    def calc_slope_sol(self, idn: np.ndarray, isky: np.ndarray):

        # 直達日射量
        self.i_d = self.cos_theta * idn

        # 天空日射量
        self.i_s = (1.0 + math.cos(self.wb)) / 2.0 * isky

# end::tagname[]

class Room:

    def __init__(self, air_volume: float, vent_volume: float):

        # 室気積[m3]
        self.air_volume = air_volume

        # 室の熱容量[J/K] 空気+家財
        self.cap = self.air_volume * 1.2 * 1005.0 + self.air_volume * 12.6 * 1000.0

        # 換気量[m3/h] →[m3/s]
        self.vent_volume = vent_volume / 3600.0

        # 部位情報
        self.building_parts = []


    def building_part_append(self, name: str, area: float, wa: float, wb: float, bp_type: str, hi: float, bp_list: list, walls: wl.Wall):

        bp_index = bp_list.index(bp_type)
        self.building_parts.append(BuildingPart(
            name,
            area,
            math.radians(wa),
            math.radians(wb),
            hi,
            walls[bp_index].rfa0,
            walls[bp_index].rft0,
            walls[bp_index].r,
            walls[bp_index].rfa1,
            walls[bp_index].rft1
        ))

        self.building_parts[-1].calc_wz_ww_ws

    def calc_d(self, a0: float, b0: float, b1: float):

        temp = ((1.0 - self.eps) * self.b0 / self.a0 + self.eps - self.c1)
        self.d0 = self.c0 / temp

        self.d1 = (1.0 - self.eps) / self.a0 / temp

        self.d2 = ()

