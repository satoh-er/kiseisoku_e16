from this import d
import pandas as pd
import numpy as np
import copy

import wall as wl
import room as rm
import weather as wdt
# import pre_calc_parameters as pcp


def main():

    # 応答係数のパラメータを読み込む
    parameters_rf = pd.read_excel('parameters.xlsx', sheet_name='parameters_rf', header=[0, 1])
    # 部位の種類名称リストを作成
    bp_list = list(set([a[0] for a in parameters_rf.columns.to_list()]))
    walls = []
    # 応答係数パラメータの読み込み
    [walls.append(wl.Wall(
        name=bp_name,
        parameter_at=parameters_rf[bp_name, 'パラメータAT'].values,
        parameter_aa=parameters_rf[bp_name, 'パラメータAA'].values,
        aa0=parameters_rf[bp_name, 'AA0'][0],
        at0=parameters_rf[bp_name, 'AT0'][0],
        eta_value=parameters_rf[bp_name, '日射熱取得率'][0],
        u_value=parameters_rf[bp_name, '熱貫流率'][0]
    )) for bp_name in bp_list]

    # 根、公比、応答係数の初項、指数項別応答係数の計算
    [wall.calc_rf() for wall in walls]

    # 空調スケジュールを読み込む
    schedule = pd.read_excel('parameters.xlsx', sheet_name='schedule', header=0)
    ac_demand = schedule['ac_demand'].values
    theta_ot_set = schedule['theta_ot,set'].values

    # 気象データを読み込む
    weather_data = wdt.Weather(pd.read_excel('parameters.xlsx', sheet_name='weather'))

    # 建物情報を読み込む
    building_info = pd.read_excel('parameters.xlsx', sheet_name='building_info', index_col=0, header=0)
    air_volume = building_info['設定値']['室気積[m3]']
    vent_volume = building_info['設定値']['換気量[m3/h]']
    beta = building_info['設定値']['放射暖房の対流比率']

    # 機器情報を読み込む
    equipment = pd.read_excel('parameters.xlsx', sheet_name='equipment', index_col=0, header=0)
    is_cooling_convection = (equipment['設定値']['冷房方式'] == '対流')
    is_heating_convection = (equipment['設定値']['暖房方式'] == '対流')
    cooling_capacity = - 1000.0 * equipment['設定値']['冷房能力[kW]']
    heating_capacity = 1000.0 * equipment['設定値']['暖房能力[kW]']

    # 室クラスの作成
    room = rm.Room(
        air_volume=air_volume,
        vent_volume=vent_volume,
        beta=beta,
        is_cooling_convenction=is_cooling_convection,
        is_heating_convvection=is_heating_convection,
        cooling_capacity=cooling_capacity,
        heating_capacity=heating_capacity
    )

    # 部位情報を読み込む
    building_parts = pd.read_excel('parameters.xlsx', sheet_name='building_parts', index_col=0)
    [room.building_part_append(
        name=name,
        area=bp['面積[m2]'],
        wa=bp['方位角[゜]'],
        wb=bp['傾斜角[゜]'],
        bp_type=bp['部位種類'],
        temp_diff_coeff=bp['温度差係数'],
        hi=bp['室内表面総合熱伝達率[W/(m2･K)]'],
        flr=bp['放射空調の発熱比率'],
        bp_list=bp_list,
        walls=walls
    ) for (name, bp) in building_parts.iteritems()]

    # 入射角の計算
    [bp.calc_cos_theta(
        sh=weather_data.sh,
        sw=weather_data.sw,
        ss=weather_data.ss
    ) for bp in room.building_parts]

    # 傾斜面日射量の計算
    [bp.calc_slope_sol(
        idn=weather_data.idn,
        isky=weather_data.isky
    ) for bp in room.building_parts]

    # 透過日射熱取得の計算（日除け、入射角特性が考慮されていない）
    q_gt_js = np.array([bp.calc_q_gt() for bp in room.building_parts])
    a_js = np.array([bp.area for bp in room.building_parts]).reshape(-1, 1)
    q_gt_ns = np.sum(a_js * q_gt_js, axis=0)

    # 各種係数の計算
    f_ot_js = room.calc_fot_js()
    f_wsr_js, f_wqr_js, f_wsrs_js, f_wscs_js, f_wqss_js = room.calc_f()
    a0 = room.calc_a0(f_wsr_js)
    eps = room.calc_eps(a0)
    b0, b1 = room.calc_b(f_wsrs_js, f_wqr_js)
    c0, c1, c2 = room.calc_c(
        k_c=0.5,
        k_r=0.5,
        f_ot_js=f_ot_js,
        f_wsr_js=f_wsr_js,
        f_wsrs_js=f_wsrs_js,
        f_wqr_js=f_wqr_js
        )
    d0, d1, d2 = room.calc_d(eps=eps, a0=a0, b0=b0, b1=b1, c0=c0, c1=c1, c2=c2)
    
    # これ以降、毎時計算
    b2 = room.calc_b2(
        f_wqss_js=f_wqr_js,
        f_wscs_js=f_wscs_js,
        theta_o_s=wdt.theta_o,
        theta_eo_s_js=wdt.theta_o,
        q_sol_s_js=wdt.isky,
        H_n=100.0
        )
    c3 = room.calc_c3(
        k_c=0.5,
        k_r=0.5,
        f_ot_js=f_ot_js,
        f_wsr_js=f_wsr_js,
        f_wscs_js=f_wscs_js,
        f_wqss_js=f_wqss_js,
        theta_eo_s_js=wdt.theta_o,
        q_sol_s_js=wdt.isky
        )
    d3 = room.calc_d3(eps=eps, a0=a0, b0=b0, b2=b2, c1=c1, c3=c3)

    print(room.calc_theta_rs(
        d0=d0,
        d1=d1,
        d2=d2,
        d3=d3,
        theta_ot_set=22.0,
        mode='H'
        ))


if __name__ == '__main__':

    main()
