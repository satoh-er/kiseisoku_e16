import pandas as pd
import numpy as np
import json

import wall as wl
import room as rm
import weather
import pre_calc_parameters as pcp
import schedule as scd
import interval
import library as lib


def main(input_data: str, output_data: str):

    delta_t = 900.0

    with open(input_data, 'r', encoding='utf-8') as js:
        rd = json.load(js)

    # 気象データの生成 => weather_for_method_file.csv
    w = weather.Weather.make_weather(
        method='ees',
        file_path='expanded_amedas',
        region=6,
        itv=interval.Interval.M15
    )

    # スケジュール
    schedule = scd.Schedule.get_schedule(
        number_of_occupants='auto',
        s_name_is=[rm['schedule']['name'] for rm in rd['rooms']],
        a_floor_is=[r['floor_area'] for r in rd['rooms']]
    )

    pp, ppg = pcp.make_pre_calc_parameters(
        delta_t=delta_t,
        rd=rd,
        w=w,
        scd=schedule
    )

    # 気象条件の1時間移動平均
    theta_o_ns = lib.moving_average_ns(pp.theta_o_ns)
    q_trs_sol_is_ns = lib.moving_average_js_ns(pp.q_trs_sol_is_ns).flatten()
    theta_eo_js_ns = lib.moving_average_js_ns(pp.theta_o_eqv_js_ns).reshape(pp.n_bdry, -1)

    # 透過日射の室内表面での吸収日射量の計算[W/m2]
    # すべての部位に均等に当たると仮定
    q_s_sol_js_ns = np.tile(q_trs_sol_is_ns / np.sum(pp.a_s_js), [pp.n_bdry, 1])

    # 建物情報を読み込む
    # 室気積[m3]
    air_volume = pp.v_rm_is.flatten()[0]
    # 外気導入量（全般換気＋局所換気）[m3/s]
    vent_volume = pp.v_vent_mec_is_ns.flatten()
    vent_volume_ns = np.insert(vent_volume, 0, vent_volume[-1])
    # 放射暖房対流比率
    beta = pp.beta_h_is.flatten()[0]

    # 機器情報を読み込む
    ed_c = rd['equipments']['cooling_equipments']
    ed_h = rd['equipments']['heating_equipments']
    is_cooling_convection = (ed_c[0]['equipment_type'] == 'rac')
    is_heating_convection = (ed_h[0]['equipment_type'] == 'rac')
    
    # 冷房能力
    if is_cooling_convection:
        cooling_capacity = - ed_c[0]['property']['q_max']
    else:
        cooling_capacity = - ed_c[0]['property']['max_capacity'] * ed_c[0]['property']['area']
    
    # 暖房能力
    if is_heating_convection:
        heating_capacity = ed_h[0]['property']['q_max']
    else:
        heating_capacity = ed_h[0]['property']['max_capacity'] * ed_h[0]['property']['area']

    # 室クラスの作成
    room = rm.Room(
        air_volume=air_volume,
        vent_volume=vent_volume_ns,
        beta=beta
    )

    # 部位情報を読み込む
    for j in range(pp.n_bdry):
        room.building_part_append(
            name=pp.name_bdry_js[j].astype(str),
            area=pp.a_s_js[j,0].astype(float),
            temp_diff_coeff=pp.k_eo_js[j,0].astype(float),
            hi=(pp.h_s_c_js[j,0] + pp.h_s_r_js[j,0]).astype(float),
            flr=pp.f_flr_h_js_is[j, 0].astype(float),
            u_value=pp.simulation_u_value[j, 0],
            rfa0=pp.phi_a0_js[j,0].astype(float),
            rft0=pp.phi_t0_js[j,0].astype(float),
            r=pp.r_js_ms[j,:].flatten(),
            rfa1=pp.phi_a1_js_ms[j,:].flatten(),
            rft1=pp.phi_t1_js_ms[j,:].flatten()
        )

    # 各種係数の計算
    f_ot_js = pp.f_mrt_hum_is_js[0,:]
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
    # 内部発熱を取得
    H_ns = np.insert(pp.q_gen_is_ns, [0], pp.q_gen_is_ns[:,-1:])
    b2 = room.calc_b2(
        f_wqss_js=f_wqr_js,
        f_wscs_js=f_wscs_js,
        theta_o_s=theta_o_ns,
        theta_eo_s_js=theta_eo_js_ns,
        q_sol_s_js=q_s_sol_js_ns,
        H_n=H_ns
        )
    c3 = room.calc_c3(
        k_c=0.5,
        k_r=0.5,
        f_ot_js=f_ot_js,
        f_wsr_js=f_wsr_js,
        f_wscs_js=f_wscs_js,
        f_wqss_js=f_wqss_js,
        theta_eo_s_js=theta_eo_js_ns,
        q_sol_s_js=q_s_sol_js_ns
        )
    d3 = room.calc_d3(eps=eps, a0=a0, b0=b0, b2=b2, c1=c1, c3=c3)

    # TODO: 冷房の設定作用温度は27℃で固定になっている
    theta_rs_c = room.calc_theta_rs(
        d0=d0,
        d1=d1,
        d2=d2,
        d3=d3,
        theta_ot_set=27.0,
        l_c=cooling_capacity if is_cooling_convection else 0.0,
        l_r=cooling_capacity if not is_cooling_convection else 0.0
        )

    # TODO: 暖房の設定作用温度は20℃で固定になっている
    theta_rs_h = room.calc_theta_rs(
        d0=d0,
        d1=d1,
        d2=d2,
        d3=d3,
        theta_ot_set=20.0,
        l_c=heating_capacity if is_heating_convection else 0.0,
        l_r=heating_capacity if not is_heating_convection else 0.0
        )
    
    result = pd.DataFrame(index=pd.date_range(start='1/1/' + '1989', periods=8760 * 4 + 1, freq='15min', name='start_time'))
    result['theta_r_s_c'] = theta_rs_c
    result['theta_r_s_h'] = theta_rs_h
    result.to_excel(output_data)


if __name__ == '__main__':

    main('input_data/cb1_ac.json', 'output_data/cb1_ac.xlsx')
    main('input_data/cb1_rad.json', 'output_data/cb1_rad.xlsx')
    main('input_data/cb2_ac.json', 'output_data/cb2_ac.xlsx')
    main('input_data/cb2_rad.json', 'output_data/cb2_rad.xlsx')
    main('input_data/ldk_ac.json', 'output_data/ldk_ac.xlsx')
    main('input_data/ldk_rad.json', 'output_data/ldk_rad.xlsx')
    main('input_data/mb_ac.json', 'output_data/mb_ac.xlsx')
    main('input_data/mb_rad.json', 'output_data/mb_rad.xlsx')
