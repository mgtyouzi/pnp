# 0:未知细胞 黑色#000000    1：肿瘤细胞 红色#FF0000      2：炎症细胞 绿色#4DFF00    3：软组织细胞 蓝色#000DFF   4：坏死细胞 黄色#F2FF00    5：非肿瘤上皮细胞 橙色#FF9D00
import sqlite3
import pandas as pd
import json
import numpy as np
import shutil
import os

def output2db(pd_points, pd_classes, template_db, save_folder, label_markgroup, table_name):
    # pd_points = np.array([[1, 2]] * 10, dtype=np.float32)
    # pd_classes = np.array([[0]] * 10, dtype=np.float32)
    # template_db = r'D:\临时文件夹_备份补充\病理项目\北京宣武医院_免疫荧光\slice.db'
    # save_folder = "./db_data/1.png/"
    # label_name = {0: "阴性细胞", 1: "阳性细胞", 2: "死细胞"}
    # label_markgroup = {0: 398, 1: 397, 2: 400}

    output = np.concatenate((pd_points, pd_classes[:, np.newaxis]), axis=1)
    data = output
    Mark_label_custom = {
        "id":[],
        "position":[],
        "method":[],
        "isExport":[],
        "remark":[],
        "aiResult":[],
        "editable":[],
        "strokeColor":[],
        "fillColor":[],
        "markType":[],
        "diagnosis":[],
        "radius":[],
        "createTime":[],
        "groupId":[],
        "areaId":[],
        "dashed":[],
        "doctorDiagnosis":[]
    }
    MarkToTile_label_custom={
        "id":[],
        "markId":[],
        "tileId":[]
    }

    start_id = 1
    count_id = 0
    tile_id = 1
    contour_list = []
    type_list = []
    for row in data:
        mask = {"x":[],"y":[]}
        #add to db
        #id
        Mark_label_custom['id'].append(int(start_id))
        start_id = start_id+1
        count_id = count_id+1
        ###############################################

        MarkToTile_label_custom['id'].append(int(tile_id))
        tile_id = tile_id + 1
        MarkToTile_label_custom['markId'].append(int(start_id - 1))
        MarkToTile_label_custom['tileId'].append(71)

        MarkToTile_label_custom['id'].append(int(tile_id))
        tile_id = tile_id + 1
        MarkToTile_label_custom['markId'].append(int(start_id - 1))
        MarkToTile_label_custom['tileId'].append(156)

        MarkToTile_label_custom['id'].append(int(tile_id))
        tile_id = tile_id + 1
        MarkToTile_label_custom['markId'].append(int(start_id - 1))
        MarkToTile_label_custom['tileId'].append(181)

        MarkToTile_label_custom['id'].append(int(tile_id))
        tile_id = tile_id + 1
        MarkToTile_label_custom['markId'].append(int(start_id - 1))
        MarkToTile_label_custom['tileId'].append(188)

        for i in range(8):
            MarkToTile_label_custom['id'].append(int(tile_id))
            tile_id = tile_id + 1
            MarkToTile_label_custom['markId'].append(int(start_id - 1))
            MarkToTile_label_custom['tileId'].append(191 + int(i))

        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 192
        #
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 193
        #
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 194
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 195
        #
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 196
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 197
        # MarkToTile_label_custom['id'].append(int(tile_id))
        # tile_id = tile_id + 1
        # MarkToTile_label_custom['markId'].append(int(start_id - 1))
        # MarkToTile_label_custom['tileId'] = 198


        ###############################################################
        #position
        value1 = row[0]  # 假设row[0]是一个浮点数
        formatted_value1 = "{:.13f}".format(value1)
        value2 = row[1]  # 假设row[0]是一个浮点数
        formatted_value2 = "{:.13f}".format(value2)
        mask['x'].append(formatted_value1)
        mask['y'].append(formatted_value2)
        mask_json = json.dumps(mask)
        Mark_label_custom['position'].append(mask_json)
        #groupId
        # Mark_label_custom['groupId'].append(int(490))
        # Mark_label_custom['groupId'].append(label_markgroup[int(row[2])])
        Mark_label_custom['groupId'].append("["+str(label_markgroup[int(row[2])])+"]")
        # Mark_label_custom['doctorDiagnosis'].append(json.dumps(None))
        Mark_label_custom['doctorDiagnosis'].append(np.nan)
        Mark_label_custom['fillColor'].append("#FF0000")
    #method
    Mark_label_custom['method'] = ["spot" for num in range(count_id)]
    # Mark_label_custom['remark'] = ["test" for num in range(count_id)]
    Mark_label_custom['remark'] = [np.nan for num in range(count_id)]

    Mark_label_custom['radius'] = [float(4.13151436527545) for num in range(count_id)]
    Mark_label_custom['createTime'] = [float(1696668185999.0 + num) for num in range(count_id)]
    #markType
    Mark_label_custom['markType'] = [1 for num in range(count_id)]
    # Mark_label_custom['markType'] = [3 for num in range(count_id)]
    #createTime
    # Mark_label_custom['createTime'] = ["1692622993151.0" for num in range(start_id)]


    null_list = [np.nan for num in range(count_id)]

    Mark_label_custom['isExport'] = null_list

    Mark_label_custom['aiResult'] = null_list
    Mark_label_custom['editable'] = null_list
    Mark_label_custom['strokeColor'] = null_list
    Mark_label_custom['diagnosis'] = null_list

    Mark_label_custom['areaId'] = null_list
    Mark_label_custom['dashed'] = null_list
    df_Mark_label_custom = pd.DataFrame(Mark_label_custom)

    df_Mark_label_custom['isExport']=df_Mark_label_custom['isExport'].astype(pd.Int64Dtype())
    df_Mark_label_custom['editable']=df_Mark_label_custom['editable'].astype(pd.Int64Dtype())
    df_Mark_label_custom['diagnosis']=df_Mark_label_custom['diagnosis'].astype(pd.Int64Dtype())
    df_Mark_label_custom['areaId']=df_Mark_label_custom['areaId'].astype(pd.Int64Dtype())
    df_Mark_label_custom['dashed']=df_Mark_label_custom['dashed'].astype(pd.Int64Dtype())

    # 将列转换为 FLOAT
    # df_Mark_label_custom['createTime'] = pd.to_numeric(df_Mark_label_custom['createTime'] , errors='coerce', downcast='float')
    # df_Mark_label_custom['radius'] = pd.to_numeric(df_Mark_label_custom['radius'] , errors='coerce', downcast='float')

    #df_Mark_label_custom['strokeColor'] = df_Mark_label_custom['strokeColor'].astype(str)
    df_MarkToTile_label_custom = pd.DataFrame(MarkToTile_label_custom)

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "slice.db")
    shutil.copyfile(template_db, save_path)
    conn = sqlite3.connect(save_path)

    Mark_table_name = "Mark_label_" + table_name
    MarkToTile_table_name = "MarkToTile_label_" + table_name

    df_Mark_label_custom.to_sql(Mark_table_name, conn, if_exists="replace", index=False,dtype={'radius': 'FLOAT','createTime':'FLOAT','aiResult':'TEXT','strokeColor':'TEXT', 'groupId':'JSON'})
    df_MarkToTile_label_custom.to_sql(MarkToTile_table_name, conn, if_exists="replace",index=False)

if __name__ == "__main__":
    # # create db file, pd_points为(n,2)的ndarray, pd_classes为(n, )的ndarray
    # from np2db import output2db
    # template_db = r'./slice.db'
    # db_save_root = "./db_data/"
    # save_folder = os.path.join(db_save_root, img_name)
    # label_name = {0: "阴性细胞", 1: "阳性细胞", 2: "死细胞"}
    # label_markgroup = {0: 398, 1: 397, 2: 400}
    # output2db(pd_points, pd_classes, template_db, save_folder, label_markgroup)
    table_name = "None"
    output2db(0, 0, 0, 0, 0, 0, table_name)