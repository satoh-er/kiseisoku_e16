import pandas as pd
import numpy as np

import wall as wl
import room as rm
import weather as wdt


def main():

    bp_list = [
        '木造_高断熱_外壁',
        '木造_高断熱_床',
        '木造_高断熱_屋根',
        'RC造_高断熱_外壁',
        'RC造_高断熱_床',
        'RC造_高断熱_屋根',
        '木造_低断熱_外壁',
        '木造_低断熱_床',
        '木造_低断熱_屋根',
        'RC造_低断熱_外壁',
        'RC造_低断熱_床',
        'RC造_低断熱_屋根',
        '共通_高断熱_窓',
        '共通_低断熱_窓'
    ]

    # 応答係数のパラメータを読み込む
    parameters_rf = pd.read_excel('parameters.xlsx', sheet_name='parameters_rf', header=[0, 1])
    walls = []
    # 応答係数パラメータの読み込み
    [walls.append(wl.Wall(
        name=bp_name,
        parameter_at=parameters_rf[bp_name, 'パラメータAT'].values,
        parameter_aa=parameters_rf[bp_name, 'パラメータAA'].values,
        aa0=parameters_rf[bp_name, 'AA0'][0],
        at0=parameters_rf[bp_name, 'AT0'][0],
        eta=parameters_rf[bp_name, '日射熱取得率'][0]
    )) for bp_name in bp_list]

    # 根、公比、応答係数の初項、指数項別応答係数の計算
    [wall.calc_rf() for wall in walls]

    print(walls[bp_list.index('共通_低断熱_窓')].aa0)

    # 空調スケジュールを読み込む
    ac_demand = pd.read_excel('parameters.xlsx', sheet_name='schedule')

    # 気象データを読み込む
    weather_data = wdt.Weather(pd.read_excel('parameters.xlsx', sheet_name='weather'))

    # 建物情報を読み込む
    building_info = pd.read_excel('parameters.xlsx', sheet_name='building_info', index_col=0, header=0)
    air_volume = building_info['設定値']['室気積[m3]']
    vent_volume = building_info['設定値']['換気量[m3/h]']
    # 室クラスの作成
    room = rm.Room(air_volume=air_volume, vent_volume=vent_volume)

    # 部位情報を読み込む
    building_parts = pd.read_excel('parameters.xlsx', sheet_name='building_parts', index_col=0)
    [room.building_part_append(
        name=name,
        area=bp['面積[m2]'],
        wa=bp['方位角[゜]'],
        wb=bp['傾斜角[゜]'],
        bp_type=bp['部位種類'],
        hi=bp['室内表面総合熱伝達率[W/(m2･K)]'],
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

    print(room.building_parts[1].area)
    print(room.building_parts[4].rfa1)


if __name__ == '__main__':
    main()
