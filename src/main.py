import pandas as pd
import numpy as np

import wall as wl


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
        parameter_at=parameters_rf['木造_高断熱_外壁', 'パラメータAT'].values,
        parameter_aa=parameters_rf['木造_高断熱_外壁', 'パラメータAA'].values,
        aa0=parameters_rf['木造_高断熱_外壁', 'AA0'][0],
        at0=parameters_rf['木造_高断熱_外壁', 'AT0'][0]
    )) for bp_name in bp_list]

    # 根、公比、応答係数の初項、指数項別応答係数の計算
    [wall.calc_rf() for wall in walls]

    print(walls[bp_list.index('共通_低断熱_窓')].aa0)

    # 空調スケジュールを読み込む
    ac_demand = pd.read_excel('parameters.xlsx', sheet_name='schedule')

    # 気象データを読み込む
    weather_data = pd.read_excel('parameters.xlsx', sheet_name='weather')

    # 建物情報を読み込む
    building_info = pd.read_excel('parameters.xlsx', sheet_name='building_info', index_col=0, header=0)
    air_volume = building_info['設定値']['室気積[m3]']
    vent_volume = building_info['設定値']['換気量[m3/h]']
    print(air_volume)
    print(vent_volume)

    # 部位情報を読み込む
    building_parts = pd.read_excel('parameters.xlsx', sheet_name='building_parts', index_col=0)
    print(building_parts)


if __name__ == '__main__':
    main()
