# import os
# import random
# import shutil

# # 原始数据路径
# image_dir = "autodl-fs/20231206@西安四院印戒细胞/第二批石蜡印戒细胞复核/image"
# json_dir = "autodl-fs/20231206@西安四院印戒细胞/第二批石蜡印戒细胞复核/json"

# # 目标数据集路径
# dataset_dir = "autodl-tmp/p2p-wq/home/zhangwanqi/code/code_cell_p2p/datasets/src-both_0401"
# train_image_dir = os.path.join(dataset_dir, "train_image")
# train_point_dir = os.path.join(dataset_dir, "train_point")
# test_image_dir = os.path.join(dataset_dir, "test_image")
# test_point_dir = os.path.join(dataset_dir, "test_point")

# # 创建目标文件夹
# os.makedirs(train_image_dir, exist_ok=True)
# os.makedirs(train_point_dir, exist_ok=True)
# os.makedirs(test_image_dir, exist_ok=True)
# os.makedirs(test_point_dir, exist_ok=True)

# # 获取所有图片文件名（不带扩展名）
# image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

# # 随机打乱文件列表
# random.shuffle(image_files)

# # 划分训练集和测试集（7:3）
# split_ratio = 0.7
# split_index = int(len(image_files) * split_ratio)
# train_files = image_files[:split_index]
# test_files = image_files[split_index:]

# # 复制训练集
# for file in train_files:
#     # 图片文件
#     src_image_path = os.path.join(image_dir, file + ".jpg")
#     dst_image_path = os.path.join(train_image_dir, file + ".jpg")
#     shutil.copy(src_image_path, dst_image_path)

#     # 标注文件
#     src_json_path = os.path.join(json_dir, file + ".jpg.json")
#     dst_json_path = os.path.join(train_point_dir, file + ".jpg.json")
#     shutil.copy(src_json_path, dst_json_path)

# # 复制测试集
# for file in test_files:
#     # 图片文件
#     src_image_path = os.path.join(image_dir, file + ".jpg")
#     dst_image_path = os.path.join(test_image_dir, file + ".jpg")
#     shutil.copy(src_image_path, dst_image_path)

#     # 标注文件
#     src_json_path = os.path.join(json_dir, file + ".jpg.json")
#     dst_json_path = os.path.join(test_point_dir, file + ".jpg.json")
#     shutil.copy(src_json_path, dst_json_path)

# print("数据集划分完成！")
# print(f"训练集数量: {len(train_files)}")
# print(f"测试集数量: {len(test_files)}")

import os
import random
import shutil

# 原始数据路径
image_dir = "autodl-fs/20231206@西安四院印戒细胞/第二批冰冻印戒细胞复核/image"
json_dir = "autodl-fs/20231206@西安四院印戒细胞/第二批冰冻印戒细胞复核/json"

# 目标数据集路径
dataset_dir = "autodl-tmp/p2p-wq/home/zhangwanqi/code/code_cell_p2p/datasets/src-both"
train_image_dir = os.path.join(dataset_dir, "train_image")
train_point_dir = os.path.join(dataset_dir, "train_point")
test_image_dir = os.path.join(dataset_dir, "test_image")
test_point_dir = os.path.join(dataset_dir, "test_point")

# 创建目标文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_point_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_point_dir, exist_ok=True)

# 获取所有图片文件名（不带扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")]

# 随机打乱文件列表
random.shuffle(image_files)

# 划分训练集和测试集（7:3）
split_ratio = 0.7
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# 统计同名文件和跳过的文件数量
duplicate_count = 0
skipped_count = 0

# 复制训练集
for file in train_files:
    # 图片文件
    src_image_path = os.path.join(image_dir, file + ".jpg")
    dst_image_path = os.path.join(train_image_dir, file + ".jpg")
    
    # 检查目标文件夹中是否已存在同名文件
    if os.path.exists(dst_image_path):
        # 添加后缀以避免覆盖
        i = 1
        while os.path.exists(os.path.join(train_image_dir, f"{file}_bingdong_{i}.jpg")):
            i += 1
        dst_image_path = os.path.join(train_image_dir, f"{file}_bingdong_{i}.jpg")
        duplicate_count += 1
    
    # 检查源文件是否存在
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
    else:
        skipped_count += 1

    # 标注文件
    src_json_path = os.path.join(json_dir, file + ".jpg.json")
    dst_json_path = os.path.join(train_point_dir, file + ".jpg.json")
    
    # 检查目标文件夹中是否已存在同名文件
    if os.path.exists(dst_json_path):
        # 添加后缀以避免覆盖
        i = 1
        while os.path.exists(os.path.join(train_point_dir, f"{file}_bingdong_{i}.jpg.json")):
            i += 1
        dst_json_path = os.path.join(train_point_dir, f"{file}_bingdong_{i}.jpg.json")
        duplicate_count += 1
    
    # 检查源文件是否存在
    if os.path.exists(src_json_path):
        shutil.copy(src_json_path, dst_json_path)
    else:
        skipped_count += 1

# 复制测试集
for file in test_files:
    # 图片文件
    src_image_path = os.path.join(image_dir, file + ".jpg")
    dst_image_path = os.path.join(test_image_dir, file + ".jpg")
    
    # 检查目标文件夹中是否已存在同名文件
    if os.path.exists(dst_image_path):
        # 添加后缀以避免覆盖
        i = 1
        while os.path.exists(os.path.join(test_image_dir, f"{file}_bingdong_{i}.jpg")):
            i += 1
        dst_image_path = os.path.join(test_image_dir, f"{file}_bingdong_{i}.jpg")
        duplicate_count += 1
    
    # 检查源文件是否存在
    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
    else:
        skipped_count += 1

    # 标注文件
    src_json_path = os.path.join(json_dir, file + ".jpg.json")
    dst_json_path = os.path.join(test_point_dir, file + ".jpg.json")
    
    # 检查目标文件夹中是否已存在同名文件
    if os.path.exists(dst_json_path):
        # 添加后缀以避免覆盖
        i = 1
        while os.path.exists(os.path.join(test_point_dir, f"{file}_bingdong_{i}.jpg.json")):
            i += 1
        dst_json_path = os.path.join(test_point_dir, f"{file}_bingdong_{i}.jpg.json")
        duplicate_count += 1
    
    # 检查源文件是否存在
    if os.path.exists(src_json_path):
        shutil.copy(src_json_path, dst_json_path)
    else:
        skipped_count += 1

# 统计目标文件夹中的文件数量
def count_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    file_count = 0
    for item in files:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count

train_image_count = count_files_in_folder(train_image_dir)
train_point_count = count_files_in_folder(train_point_dir)
test_image_count = count_files_in_folder(test_image_dir)
test_point_count = count_files_in_folder(test_point_dir)

# 输出结果
print("数据集划分完成！")
print(f"训练集数量: {len(train_files)}")
print(f"测试集数量: {len(test_files)}")
print(f"同名文件数量: {duplicate_count}")
print(f"跳过的文件数量: {skipped_count}")
print(f"训练集图片文件数量: {train_image_count}")
print(f"训练集标注文件数量: {train_point_count}")
print(f"测试集图片文件数量: {test_image_count}")
print(f"测试集标注文件数量: {test_point_count}")