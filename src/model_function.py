import glob
import math
import os
import re

import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, optimizers
from tensorflow.keras.layers import Input
from tqdm import tqdm

from data_prepare_func import *

tf.config.experimental_run_functions_eagerly(True)
INIT_POSITION = '/data/ziming/2024TCC/MGGFL_code'
LOG_POSITION = INIT_POSITION + '/log/'
tf.random.set_seed(2345)
stack_n = 18
train_num = 50000
batch_size = 64
weight_decay = 0.0005
test_batch_size = 64
warmup_steps = 2
total_steps = 20
train_lr_init = 1e-1
train_lr_end = 1e-7
local_epoch = 5
att_epoch = 30
distillate_epoch = 60

rate = math.pow(train_lr_end / train_lr_init, 1 / (total_steps * local_epoch - warmup_steps * local_epoch))

optimizer = optimizers.SGD(momentum=0.9, nesterov=True)
img_input = Input(shape=(32, 32, 3))

def set_seed(seed):
    # 为 Python 内建的随机数生成器设置种子
    random.seed(seed)  
    # 为 NumPy 设置种子
    np.random.seed(seed)   
    # 为 TensorFlow 设置种子
    tf.random.set_seed(seed)
    
def clean_log():
    # 对log文件夹进行清空
    file_paths = glob.glob('../log/small_coarse*.txt')
    for file_path in file_paths:
        with open(file_path, 'w') as f:
            f.write('')

    file_paths = glob.glob('../log/big_fine*.txt')
    for file_path in file_paths:
        with open(file_path, 'w') as f:
            f.write('')


def get_model_vars(model, attri=None):
    # 获得模型参数
    new_models_vars = {}
    if attri == 'small_coarse':  # 小模型+粗粒度
        for i in range(22):
            conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
            batch_layer_name = 'batch_normalization_%d' % i if i > 0 else 'batch_normalization'
            if i < 19:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights(), dtype=object)
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights(), dtype=object)
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights(), dtype=object)
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights(), dtype=object)
        new_models_vars[dense_layer_name] = dense_layer_vars

    elif attri == 'big_fine':  # 大模型+细粒度
        for i in range(28):
            conv_layer_name = 'conv2d_%d' % int(22 + i)
            batch_layer_name = 'batch_normalization_%d' % int(19 + i)
            if i < 25:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights(), dtype=object)
                new_models_vars[conv_layer_name] = conv_layer_vars
                batch_layer_vars = np.array(model.get_layer(batch_layer_name).get_weights(), dtype=object)
                new_models_vars[batch_layer_name] = batch_layer_vars
            else:
                conv_layer_vars = np.array(model.get_layer(conv_layer_name).get_weights(), dtype=object)
                new_models_vars[conv_layer_name] = conv_layer_vars
        dense_layer_name = 'dense_1'
        dense_layer_vars = np.array(model.get_layer(dense_layer_name).get_weights(), dtype=object)
        new_models_vars[dense_layer_name] = dense_layer_vars

    return new_models_vars


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def l2_loss(model, weights=weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


def cross_entropy(y_true, y_pred):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(cross_entropy)


def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    # accura = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    accura_num = tf.reduce_sum(tf.cast(correct_num, dtype=tf.float32))
    return accura_num


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, prediction


@tf.function
def test_step(model, x, y):
    prediction = model(x, training=False)
    ce = cross_entropy(y, prediction)
    return ce, prediction


def fine_coarse_accuracy(M1, y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accura = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))

    prediction = np.array(y_pred)
    y = np.array(y_true)
    y = tf.matmul(y, M1)
    y = np.array(y).squeeze()
    y = tf.cast(y, tf.float32)

    b = np.zeros(prediction.shape)
    b[np.arange(len(prediction)), prediction.argmax(1)] = 1
    b = tf.cast(b, tf.float32)
    prediction1 = tf.matmul(b, M1)
    prediction1 = np.array(prediction1).squeeze()
    prediction1 = tf.cast(prediction1, tf.float32)
    fine_coarse_accu = accuracy(y, prediction1)

    return accura, fine_coarse_accu


def train(model, optimizer, images, labels, lr):
    sum_loss = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)
    images = images_augment(images)
    train_db = tf.data.Dataset.from_tensor_slices((images, labels))
    train_db = train_db.shuffle(10000).map(preprocess, num_parallel_calls=2).batch(batch_size)
    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    optimizer.lr.assign(lr)
    for (x, y) in tqdm(train_db):
        loss, prediction = train_step(model, optimizer, x, y)
        sum_loss += loss


def test(model, images, labels, order, lr):
    sum_loss = 0
    sum_accuracy = 0
    test_db = tf.data.Dataset.from_tensor_slices((images, labels))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for (x, y) in tqdm(test_db):
        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)
    all_numbers = int(len(labels))
    print(all_numbers)
    sum_loss = sum_loss / int(all_numbers / test_batch_size)
    # sum_loss = sum_loss / int(all_numbers)
    sum_accuracy = sum_accuracy / all_numbers
    tqdm.write('test, loss:%f, accuracy:%f' % (sum_loss, sum_accuracy))
    with open(LOG_POSITION + order + '.txt', 'a') as f:
        f.write(str(sum_accuracy))
        # f.write(str(lr))
        f.write('\n')
        f.close()


def learning_rate(epoch):
    if epoch < warmup_steps * local_epoch:
        lr = epoch / (warmup_steps * local_epoch) * train_lr_init
    else:
        # lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
        #     (1 + tf.cos((epoch - warmup_steps*5) / (total_steps*5 - warmup_steps*5) * np.pi))
        # )
        lr = (train_lr_init / (math.pow(rate, warmup_steps * local_epoch))) * (rate ** epoch)
        # lr=0.167*(0.95**epoch)
    return lr


def client_train(model, data, order, model_vars, global_epoch, gra_attri):
    (train_images, train_labels, test_images, test_labels) = data

    # if train_type == 'partition':
    #     # 在预训练的时候，数据量为test的0.1
    #     train_images, test_images, train_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.1)

    state_accuracy = []
    for key in model_vars:
        model.get_layer(key).set_weights(model_vars[key])
    for p in range(local_epoch):
        lr = learning_rate(global_epoch * local_epoch + p + 1)
        '''------------------------------------------'''
        train(model, optimizer, train_images, train_labels, lr=lr)
        test(model, test_images, test_labels, gra_attri + order, lr=lr)
    with open(LOG_POSITION + gra_attri + order + '.txt', 'r') as f:
        lines = f.readlines()[-local_epoch:]
        for a in lines:
            result = re.search(r'\d+(\.\d+)?', a).group()
            result = float(result)
            state_accuracy.append(result * 100)
        f.close()
    states = np.mean(state_accuracy)
    modelvars = get_model_vars(model, gra_attri)
    model.save_weights('../model/' + gra_attri + order)
    return modelvars, states


def distillate(student_vars, teacher_vars, Gv, share_data, stu_gra, tea_gra, small_coarse_output, big_fine_output):
    # 模型蒸馏，也就是指导操作
    new_student_vars = []

    if stu_gra == 'small_coarse':
        stu_output = small_coarse_output
    else:
        stu_output = big_fine_output

    if tea_gra == 'small_coarse':
        tea_output = small_coarse_output
    else:
        tea_output = big_fine_output

    for i, client_model_vars in enumerate(student_vars):

        if Gv[i]:
            tea_model = models.Model(img_input, tea_output)
            sub_tea_model = tf.keras.models.Model(inputs=tea_model.input, outputs=tea_model.layers[-2].output)
            for layer in sub_tea_model.layers:
                if layer.name in client_model_vars:
                    sub_tea_model.get_layer(layer.name).set_weights(client_model_vars[layer.name])
            del tea_model

            for num in Gv[i]:

                optimizer.lr.assign(0.1)
                stu_model = models.Model(img_input, stu_output)
                sub_stu_model = tf.keras.models.Model(inputs=stu_model.input,
                                                      outputs=stu_model.layers[-2].output)
                for layer in sub_stu_model.layers:
                    if layer.name in teacher_vars[num]:
                        sub_stu_model.get_layer(layer.name).set_weights(teacher_vars[num][layer.name])
                del stu_model

                data = (share_data[i][0], share_data[i][1])
                for j in range(distillate_epoch):
                    train_db = tf.data.Dataset.from_tensor_slices(data)
                    train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
                    train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

                    for (x, y) in tqdm(train_db):
                        '''--------------------------------------------------------------------------------'''
                        stu_y = sub_tea_model(x)
                        with tf.GradientTape() as tape:
                            pre = sub_stu_model(x, training=True)
                            loss = tf.reduce_mean(
                                tf.losses.MSE(stu_y, pre))
                            gradients = tape.gradient(loss, sub_stu_model.trainable_variables)
                            optimizer.apply_gradients(zip(gradients, sub_stu_model.trainable_variables))
                del sub_tea_model

            stu_model = models.Model(img_input, stu_output)
            for layer in sub_stu_model.layers:
                if layer.name in client_model_vars:
                    layer_vars = np.array(sub_stu_model.get_layer(layer.name).get_weights(), dtype=object)
                    stu_model.get_layer(layer.name).set_weights(layer_vars)
            del sub_stu_model
            for v in stu_model.layers[-1:]:
                v.trainable = True
            for v in stu_model.layers[:-1]:
                v.trainable = False
            optimizer.lr.assign(0.01)
            for j in range(5):
                train_db = tf.data.Dataset.from_tensor_slices(data)
                train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
                train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                for (x, y) in tqdm(train_db):
                    with tf.GradientTape() as tape:
                        prediction = stu_model(x, training=True)
                        ce = cross_entropy(y, prediction)
                        l2 = l2_loss(stu_model)
                        loss = ce + l2
                        gradients = tape.gradient(loss, stu_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, stu_model.trainable_variables))
            client_model_vars = get_model_vars(stu_model, stu_gra)
            new_student_vars.append(client_model_vars)
            for v in stu_model.layers[:-1]:
                v.trainable = True
            del stu_model
        else:
            new_student_vars.append(client_model_vars)
            continue

    return new_student_vars


def client_train_one_gra(model, vars, data, global_epoch, gra_attri):
    vars_result = []
    accs_result = []

    for i, (d, v) in enumerate(zip(data, vars)):
        var, acc = client_train(model, d, order=str(i + 1), model_vars=v, global_epoch=global_epoch,
                                gra_attri=gra_attri)
        vars_result.append(var)
        accs_result.append(acc)

    return vars_result, accs_result


def client_train_one_gra_malice(model, vars, data, global_epoch, gra_attri, malicious, malice_num, node_num):
    vars_result = []
    accs_result = []

    # 定义日志文件路径
    log_path = os.path.join(INIT_POSITION, 'log', 'var0')

    # 将恶意标志记录到日志文件
    with open(log_path, 'a') as f:
        f.write(f'Malicious: {malicious}\n' + '-----------------------------------------------------\n')

    # 确保传入的客户端数量不超过总数
    malice_num = min(malice_num, node_num)

    # 使用malicious变量的客户端索引集合
    malice_indices = set(range(malice_num))

    # 遍历所有客户端
    for i in range(node_num):
        # 判断当前客户端是否应当使用malicious变量
        current_vars = malicious if i in malice_indices else vars[i]

        # 调用client_train进行训练
        var, acc = client_train(model, data[i], order=str(i + 1), model_vars=current_vars,
                                global_epoch=global_epoch, gra_attri=gra_attri)

        # 保存每个客户端的训练结果
        vars_result.append(var)
        accs_result.append(acc)

    return vars_result, accs_result


def aggregate(all_model, credit_list):
    # 计算所有模型的权重
    total_credit = sum(credit_list)
    weights = [credit / total_credit for credit in credit_list]

    # 计算每层权重加权后的新参数
    new_model_vars = []

    select_model = all_model[0]
    for key in select_model:
        layer = 0
        for j, another_model in enumerate(all_model):
            layer += weights[j] * another_model[key]
        select_model[key] = layer
    new_model_vars = [select_model for i in range(len(all_model))]

    return new_model_vars


def model_guidance(M, tea_model, big_fine_vars, small_coarse_share_data, small_coarse_acc, tea_index, stu_index):
    # 一对一教导参数获得

    pi = 0.0
    m1 = M.astype('float32')

    data = small_coarse_share_data[stu_index]
    fine_var = big_fine_vars[tea_index - 15]
    for key in fine_var:
        tea_model.get_layer(key).set_weights(fine_var[key])
    (images, labels) = data
    test_db = tf.data.Dataset.from_tensor_slices((images, labels))
    test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
    test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    sum_accuracy = 0.0
    for (x, y) in tqdm(test_db):
        prediction = tea_model(x, training=False)
        prediction = np.array(prediction)
        b = np.zeros(prediction.shape)
        b[np.arange(len(prediction)), prediction.argmax(1)] = 1
        b = tf.cast(b, tf.float32)
        prediction1 = tf.matmul(b, m1)
        prediction1 = np.array(prediction1).squeeze()
        prediction1 = tf.cast(prediction1, tf.float32)
        accu = accuracy(y, prediction1)
        sum_accuracy += accu
    sum_accuracy = sum_accuracy / int(len(labels))
    pi = sum_accuracy * 100

    threshold = small_coarse_acc[stu_index]

    print("edge({}, {})'s sum_accuracy is {}, threshold is {}".format(tea_index, stu_index, sum_accuracy, threshold))

    return pi


def cross_relationship_partition(M, fine_model, big_fine_vars, coarse_share_data, acc, name, nodes):
    # graph partition时候采用的函数
    # 获得指导参数矩阵
    L = np.zeros([len(coarse_share_data), len(big_fine_vars)], np.float32)
    Gv = np.zeros([len(coarse_share_data), len(big_fine_vars)], np.float32)

    M1 = M.astype('float32')

    for i, data in enumerate(coarse_share_data):
        # i为student， j为teacher

        for j, coarse_vars in enumerate(big_fine_vars):
            for key in coarse_vars:
                fine_model.get_layer(key).set_weights(coarse_vars[key])
            (images, labels) = data
            test_db = tf.data.Dataset.from_tensor_slices((images, labels))
            test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
            test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            sum_accuracy = 0
            for (x, y) in tqdm(test_db):
                prediction = fine_model(x, training=False)
                prediction = np.array(prediction)
                b = np.zeros(prediction.shape)
                b[np.arange(len(prediction)), prediction.argmax(1)] = 1
                b = tf.cast(b, tf.float32)
                prediction1 = tf.matmul(b, M1)
                prediction1 = np.array(prediction1).squeeze()
                prediction1 = tf.cast(prediction1, tf.float32)
                accu = accuracy(y, prediction1)
                sum_accuracy += accu
            sum_accuracy = sum_accuracy / int(len(labels))

            L[i, j] = sum_accuracy * 100
            Gv[i, j] = sum_accuracy * 100 - acc[i]

    data1 = pd.DataFrame(Gv)
    data1.to_csv(LOG_POSITION + f'Gv_{name}_{nodes}' + '.csv', mode='w', header=False, index=False)
    return Gv


def cross_relationship(M, model, model_vars0, share_data, acc, cross_num):
    # 正式训练时候需要的函数
    L = np.zeros([len(share_data), len(model_vars0)], np.float32)
    Gv = []

    M1 = M.astype('float32')
    # M2=M[1].astype('float32')
    for i, data in enumerate(share_data):
        # i为student， j为teacher
        for j, coarse_vars in enumerate(model_vars0):
            for key in coarse_vars:
                model.get_layer(key).set_weights(coarse_vars[key])
            (images, lables) = data
            test_db = tf.data.Dataset.from_tensor_slices((images, lables))
            test_db = test_db.map(preprocess, num_parallel_calls=2).batch(test_batch_size)
            test_db = test_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            sum_accuracy = 0
            for (x, y) in tqdm(test_db):
                prediction = model(x, training=False)
                prediction = np.array(prediction)
                b = np.zeros(prediction.shape)
                b[np.arange(len(prediction)), prediction.argmax(1)] = 1
                b = tf.cast(b, tf.float32)
                prediction1 = tf.matmul(b, M1)
                prediction1 = np.array(prediction1).squeeze()
                prediction1 = tf.cast(prediction1, tf.float32)
                accu = accuracy(y, prediction1)
                sum_accuracy += accu
            sum_accuracy = sum_accuracy / int(len(lables))

            L[i, j] = sum_accuracy * 100
        threshold = acc[i]
        a = np.argmax(L[i])
        # b = np.max(L[i])
        b = []
        weight = np.flatnonzero(L[i] >= threshold).tolist()

        if weight:
            b.append(int(a))
            Gv.append(b)
        else:
            Gv.append([])
    # data1 = pd.DataFrame(Gv)
    # data1.to_csv('Gv' + str(cross_num) + '.csv', mode='a', header=False, index=False)
    return Gv

# def partition_distillate(student_var, teacher_var, gv, share_data, small_coarse_output, big_fine_output, stu_index,
#                          tea_index):
#     # 计算单个的指导效果, index为1-30
#     stu_index -= 1
#     tea_index -= 1
#
#     new_student_var = []
#     stu_output = small_coarse_output
#     tea_output = big_fine_output
#
#     stu_gra = 'small_coarse'
#     tea_gra = 'big_fine'
#
#     if gv[stu_index]:
#         tea_model = models.Model(img_input, tea_output)
#         sub_tea_model = tf.keras.models.Model(inputs=tea_model.input, outputs=tea_model.layers[-2].output)
#         for layer in sub_tea_model.layers:
#             if layer.name in student_var:
#                 sub_tea_model.get_layer(layer.name).set_weights(student_var[layer.name])
#             del tea_model
#
#         optimizer.lr.assign(0.1)
#         stu_model = models.Model(img_input, stu_output)
#         sub_stu_model = tf.keras.models.Model(inputs=stu_model.input, outputs=stu_model.layers[-2].output)
#
#         for layer in sub_stu_model.layers:
#             if layer.name in teacher_var:
#                 sub_stu_model.get_layer(layer.name).set_weights(teacher_var[layer.name])
#         del stu_model
#
#         data = (share_data[stu_index][0], share_data[stu_index][1])
#
#         for j in range(distillate_epoch):
#             train_db = tf.data.Dataset.from_tensor_slices(data)
#             train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
#             train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#
#             for (x, y) in tqdm(train_db):
#                 '''--------------------------------------------------------------------------------'''
#                 stu_y = sub_tea_model(x)
#                 with tf.GradientTape() as tape:
#                     pre = sub_stu_model(x, training=True)
#                     loss = tf.reduce_mean(
#                         tf.losses.MSE(stu_y, pre))
#                     gradients = tape.gradient(loss, sub_stu_model.trainable_variables)
#                     optimizer.apply_gradients(zip(gradients, sub_stu_model.trainable_variables))
#         del sub_tea_model
#
#         stu_model = models.Model(img_input, stu_output)
#         for layer in sub_stu_model.layers:
#             if layer.name in student_var:
#                 layer_vars = np.array(sub_stu_model.get_layer(layer.name).get_weights(), dtype=object)
#                 stu_model.get_layer(layer.name).set_weights(layer_vars)
#         del sub_stu_model
#         for v in stu_model.layers[-1:]:
#             v.trainable = True
#         for v in stu_model.layers[:-1]:
#             v.trainable = False
#         optimizer.lr.assign(0.01)
#         # FIXME: 为什么这里要弄五次？
#         for j in range(5):
#             train_db = tf.data.Dataset.from_tensor_slices(data)
#             train_db = train_db.shuffle(1000).map(preprocess, num_parallel_calls=2).batch(batch_size)
#             train_db = train_db.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#             for (x, y) in tqdm(train_db):
#                 with tf.GradientTape() as tape:
#                     prediction = stu_model(x, training=True)
#                     ce = cross_entropy(y, prediction)
#                     l2 = l2_loss(stu_model)
#                     loss = ce + l2
#                     gradients = tape.gradient(loss, stu_model.trainable_variables)
#                     optimizer.apply_gradients(zip(gradients, stu_model.trainable_variables))
#         client_model_vars = get_model_vars(stu_model, stu_gra)
#         new_student_var.append(client_model_vars)
#         for v in stu_model.layers[:-1]:
#             v.trainable = True
#         del stu_model
#     else:
#         new_student_var.append(student_var)
#
#     return new_student_var
