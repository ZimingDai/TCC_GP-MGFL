import random

import cv2
import numpy as np
import tensorflow as tf

'''-----------------这里是对cifar图像的处理--------------------'''


def image_ip(train_images, train_labels, test_images, test_labels, num_class1):
    train_labels = tf.keras.utils.to_categorical(train_labels, num_class1)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_class1)

    train_images = np.array(train_images, dtype=np.float32)
    test_images = np.array(test_images, dtype=np.float32)

    train_images, test_images = color_normalize(train_images, test_images)

    return train_images, train_labels, test_images, test_labels


def color_normalize(train_images, test_images):
    mean = [np.mean(train_images[:, :, :, i]) for i in range(3)]  # [125.307, 122.95, 113.865]
    std = [np.std(train_images[:, :, :, i]) for i in range(3)]  # [62.9932, 62.0887, 66.7048]
    for i in range(3):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / (std[i] * 255)
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / (std[i] * 255)
    return train_images, test_images


def images_augment(images):
    output = []
    for img in images:
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        if np.random.randint(0, 2):
            img = cv2.flip(img, 1)
        output.append(img[x: x + 32, y:y + 32, :])
    return np.ascontiguousarray(output, dtype=np.float32)


'''-----------------------------------------------------------'''


def create_fine_splits(y_20, x_fine, y_fine, x_test_fine, y_test_fine, fine_class, client_num, class_num):
    # 对fine100的数据处理，获得train，test，share
    dic = {}
    for i in range(20):
        name = '%d' % i
        dic[name] = []
    for i, label in enumerate(y_20):
        a = y_fine[i]
        name = '%d' % (int(label))
        dic[name].append(int(a))
    for key in dic:
        dic[key] = list(set(dic[key]))
    fine1 = []
    for i in fine_class:
        fine1 += dic[str(int(i))]
    num1 = []
    share_num1 = []
    for i in fine1:
        num11 = []
        for j, label in enumerate(y_fine):
            if i == int(label):
                num11.append(int(j))
        num112 = random.sample(num11, int(len(num11) / client_num))
        share_num11 = random.sample(num112, int(len(num112) / 2))
        num1 += num112
        share_num1 += share_num11
    random.shuffle(num1)
    random.shuffle(share_num1)
    xtrain_fine1 = []
    ytrain_fine1 = []
    x_share1 = []
    y_share1 = []
    for i in num1:
        xtrain_fine1.append(x_fine[i])
        ytrain_fine1.append(y_fine[i])
    xtrain_fine1 = np.array(xtrain_fine1)
    ytrain_fine1 = np.array(ytrain_fine1)
    for i in share_num1:
        x_share1.append(x_fine[i])
        y_share1.append(y_fine[i])
    x_share1 = np.array(x_share1)
    y_share1 = np.array(y_share1)
    '''----------------------------------------------------------------------'''
    num2 = []
    for i in fine1:
        num22 = []
        for j, label in enumerate(y_test_fine):
            if i == int(label):
                num22.append(int(j))
        # num22 = random.sample(num22, int(len(num22) / client_num))
        num2 += num22
    random.shuffle(num2)
    xtest_fine1 = []
    ytest_fine1 = []
    for i in num2:
        xtest_fine1.append(x_test_fine[i])
        ytest_fine1.append(y_test_fine[i])
    xtest_fine1 = np.array(xtest_fine1)
    ytest_fine1 = np.array(ytest_fine1)

    xtrain_fine1, ytrain_fine1, xtest_fine1, ytest_fine1 = image_ip(
        xtrain_fine1, ytrain_fine1, xtest_fine1, ytest_fine1, class_num)

    x_share1, y_share1, x_share1, y_share1 = image_ip(x_share1, y_share1, x_share1, y_share1, class_num)
    return (xtrain_fine1, ytrain_fine1, xtest_fine1, ytest_fine1), (x_share1, y_share1)


def create_coarse_splits(x_coarse, y_coarse, x_test_coarse, y_test_coarse, coarse_class, client_num, class_num):
    # 对粗粒度的数据进行拆分

    train_index = []  # num1
    share_index = []  # share_num1
    for i in coarse_class:
        # 从全部的data中提取出coarse_class，并同时获得与训练集一半数量的share集合
        select_class_index = []  # num11
        for j, label in enumerate(y_coarse):
            if i == int(label):
                select_class_index.append(int(j))
        train_i_index = random.sample(select_class_index, int(len(select_class_index) / client_num))  # num112
        share_i_index = random.sample(train_i_index, int(len(train_i_index) / 2))  # share_num11
        share_index += share_i_index
        train_index += train_i_index
    random.shuffle(train_index)
    random.shuffle(share_index)
    '''-----------------------------------------------'''

    x_train_coarse = []
    y_train_coarse = []

    for i in train_index:
        # 通过index添加x, y
        x_train_coarse.append(x_coarse[i])
        y_train_coarse.append(y_coarse[i])
    x_train_coarse = np.array(x_train_coarse)
    y_train_coarse = np.array(y_train_coarse)

    x_share_coarse = []
    y_share_coarse = []
    for i in share_index:
        # 通过index添加x, y
        x_share_coarse.append(x_coarse[i])
        y_share_coarse.append(y_coarse[i])
    x_share_coarse = np.array(x_share_coarse)
    y_share_coarse = np.array(y_share_coarse)
    '''--------------------------------------------------'''

    test_index = []  # num2
    for i in coarse_class:
        test_i_index = []  # num22
        for j, label in enumerate(y_test_coarse):
            if i == int(label):
                test_i_index.append(int(j))
        # num22 = random.sample(num22, int(len(num22) / client_num))
        test_index += test_i_index
    random.shuffle(test_index)

    x_test = []
    y_test = []
    for i in test_index:
        x_test.append(x_test_coarse[i])
        y_test.append(y_test_coarse[i])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train_coarse, y_train_coarse, x_test, y_test = image_ip(x_train_coarse, y_train_coarse,
                                                              x_test, y_test, class_num)
    x_share_coarse, y_share_coarse, x_share_coarse, y_share_coarse = image_ip(x_share_coarse, y_share_coarse,
                                                                              x_share_coarse, y_share_coarse, class_num)

    return (x_train_coarse, y_train_coarse, x_test, y_test), (x_share_coarse, y_share_coarse)


def create_datasets(num_small, num_big, x_coarse, y_coarse, x_fine, y_fine, x_test_coarse, y_test_coarse, x_test_fine,
                    y_test_fine, coarse_class, fine_class, coarse_class_num, fine_class_num):
    small_coarse_data_list = []
    small_coarse_share_list = []
    big_fine_data_list = []
    big_fine_share_list = []

    # 创建小模型的数据集
    for _ in range(num_small):
        small_coarse_data, small_coarse_share = create_coarse_splits(x_coarse, y_coarse, x_test_coarse, y_test_coarse,
                                                                     coarse_class, 7, coarse_class_num)
        small_coarse_data_list.append(small_coarse_data)
        small_coarse_share_list.append(small_coarse_share)

    # 创建大模型的数据集
    for _ in range(num_big):
        big_fine_data, big_fine_share = create_fine_splits(y_coarse, x_fine, y_fine, x_test_fine, y_test_fine,
                                                           fine_class, 7, fine_class_num)
        big_fine_data_list.append(big_fine_data)
        big_fine_share_list.append(big_fine_share)
    print("Creating dataset finished.")
    return small_coarse_data_list, small_coarse_share_list, big_fine_data_list, big_fine_share_list


def create_coarse_data(x_fine, y_fine, x_test_fine, y_test_fine):
    # 定义映射关系
    mapping = {
        0: 0,
        9: 1, 5: 1,
        3: 2, 2: 2, 8: 2,
        6: 3,
        4: 4,
        1: 5, 7: 5
    }

    # 定义函数用于转换细粒度标签到粗粒度标签
    def map_labels(y_fine, mapping):
        # 创建一个与 y_fine 形状相同，但填充为零的数组
        y_coarse = np.zeros_like(y_fine)
        # 遍历映射关系，更新粗粒度标签
        for fine_label, coarse_label in mapping.items():
            y_coarse[y_fine == fine_label] = coarse_label
        return y_coarse

    # 转换训练集和测试集标签
    y_coarse = map_labels(y_fine, mapping)
    y_test_coarse = map_labels(y_test_fine, mapping)

    print("Creating coarse data finished.")
    return (x_fine, y_coarse), (x_test_fine, y_test_coarse)


def adjust_image_shape_and_channels(images):
    resized_images = []

    # 对每个图像进行调整
    for img in images:
        # 调整图像大小为 32x32
        # cv2.resize 需要明确的两维图像，因此如果图像是灰度的，不需要额外的维度
        resized_img = cv2.resize(img, (32, 32))

        # 将灰度图像转换为三通道图像（通过堆叠单通道三次）
        rgb_img = np.stack([resized_img] * 3, axis=-1)

        resized_images.append(rgb_img)

    # 将列表转换回 numpy 数组
    resized_images = np.array(resized_images)
    return resized_images
