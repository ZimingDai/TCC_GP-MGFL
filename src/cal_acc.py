import argparse
import copy
import csv
import math
import os
import pickle
import sys

import tensorflow as tf
from tensorflow.keras import backend

from model_function import *
from prepare_csv_func import *
from model import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def csv_reader(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        a = [[eval(item) for item in row] for row in reader]
        f.close()
    return a


if __name__ == '__main__':
    backend.clear_session()
    clean_log()

    '''-----------超参数定义--------------'''
    parser = argparse.ArgumentParser()

    # 构建图点与边权重的超参数
    parser.add_argument("--nodes", default=30, type=int,
                        help="Total number of edge devices, 50/50 coarse and fine")

    parser.add_argument("--malice", default=0, type=int,
                        help="Number of malicious nodes")

    parser.add_argument("--device", default="cpu", type=str,
                        help="Choose the device to train (now only can use cuda:0). eg: 0, default is cpu")

    parser.add_argument("--epoch", default=total_steps, type=int,
                        help="The number of epochs")

    parser.add_argument("--k", default=2, type=int, help="Group number")

    parser.add_argument("--dataset", default="cifar100", type=str, help="Dataset name")

    parser.add_argument("--coarse_class", required=True, type=int,
                        help="Number of categories in a coarse-grained dataset")

    parser.add_argument("--fine_class", required=True, type=int, help="Number of categories in a fine-grained dataset")

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    
    parser.add_argument("--method", default="MGGFL", type=str,
                        help="What method to cluster? eg. MGGFL, Greedy, Random, Spectral.")

    args = parser.parse_args()
    
    set_seed(args.seed)

    if args.device != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        tf.config.experimental_run_functions_eagerly(True)
        devices = tf.config.experimental.list_physical_devices('GPU')
        print("useable device:", devices)
        tf.config.experimental.set_memory_growth(devices[0], True)

    '''---------------获取图的数据与图分割之后的数据--------------'''
    with open(INIT_POSITION + '/data/graph.gpickle', 'rb') as f:
        G = pickle.load(f)
        f.close()

    file_path = f'../data/node_partition_{args.dataset}_{args.nodes}_{args.k}_{args.method}.csv'
    if os.path.exists(file_path):
        nodes_blocks_index_list = csv_reader(file_path)
    else:
        print("No graph partition data available, please run graph_partition.py")
        sys.exit()  

    '''--------------------模型相互指导参数获得--------------------'''
    coarse_class = [0, 1, 2, 3, 4, 5]
    fine_class = coarse_class

    if args.dataset == "cifar100":
        (x_coarse, y_coarse), (x_test_coarse, y_test_coarse) = tf.keras.datasets.cifar100.load_data('coarse')
        (x_fine, y_fine), (x_test_fine, y_test_fine) = tf.keras.datasets.cifar100.load_data('fine')
    elif args.dataset == "mnist":
        (x_fine, y_fine), (x_test_fine, y_test_fine) = tf.keras.datasets.mnist.load_data()
        (x_coarse, y_coarse), (x_test_coarse, y_test_coarse) = create_coarse_data(x_fine, y_fine, x_test_fine,
                                                                                  y_test_fine)
        x_fine = adjust_image_shape_and_channels(x_fine)
        x_test_fine = adjust_image_shape_and_channels(x_test_fine)
        x_coarse = adjust_image_shape_and_channels(x_coarse)
        x_test_coarse = adjust_image_shape_and_channels(x_test_coarse)
        print("Adjusting images finished.")

    num_small = args.nodes // 2
    num_big = args.nodes // 2

    small_coarse_data_list, small_coarse_share_list, big_fine_data_list, big_fine_share_list = create_datasets(
        num_small, num_big, x_coarse, y_coarse, x_fine, y_fine, x_test_coarse, y_test_coarse, x_test_fine, y_test_fine,
        coarse_class, fine_class, args.coarse_class, args.fine_class)

    # 模型的初始化，由于初始化相同，所以所有节点共用一个初始化器

    small_coarse_output = make_small_resnet_filter(img_input, model_size=args.coarse_class)
    small_coarse_model = models.Model(img_input, small_coarse_output)
    small_coarse_model.summary()
    small_coarse_var = get_model_vars(small_coarse_model, attri='small_coarse')
    del small_coarse_model

    big_fine_output = make_big_resnet_filter(img_input, model_size=args.fine_class)
    big_fine_model = models.Model(img_input, big_fine_output)
    big_fine_model.summary()
    big_fine_var = get_model_vars(big_fine_model, attri='big_fine')
    del big_fine_model

    # 计算每一个模型的初始credit
    small_credit_list = []
    small_total_data_len = sum([len(data) for data in small_coarse_data_list])
    for data in small_coarse_data_list:
        small_credit_list.append(len(data) / small_total_data_len)

    big_credit_list = []
    big_total_data_len = sum([len(data) for data in big_fine_data_list])
    for data in big_fine_data_list:
        big_credit_list.append(len(data) / big_total_data_len)

    small_coarse_vars = [small_coarse_var for i in range(num_small)]
    big_fine_vars = [big_fine_var for i in range(num_big)]

    '''--------------对share_data, vars, credit等进行分组--------------'''

    # 分组后的数据

    # 训练数据
    small_coarse_data_grouped = []
    big_fine_data_grouped = []
    # share数据
    small_coarse_share_data_grouped = []
    big_fine_share_data_grouped = []
    # credit 数据
    small_credit_list_grouped = []
    big_credit_list_grouped = []

    # 获得节点的一维数组，用来进行排序
    s_nodes_blocks_index_list_flatten = []
    b_nodes_blocks_index_list_flatten = []

    for group in nodes_blocks_index_list:

        small_coarse_data = []
        big_fine_data = []

        small_coarse_share_data = []
        big_fine_share_data = []

        single_small_credit_list = []
        single_big_credit_list = []

        for index in group:
            if index < num_small:
                small_coarse_data.append(small_coarse_data_list[index])
                small_coarse_share_data.append(small_coarse_share_list[index])
                single_small_credit_list.append(small_credit_list[index])
                s_nodes_blocks_index_list_flatten.append(index)
            else:
                big_fine_data.append(big_fine_data_list[index - num_small])
                big_fine_share_data.append(big_fine_share_list[index - num_small])
                single_big_credit_list.append(big_credit_list[index - num_small])
                b_nodes_blocks_index_list_flatten.append(index - num_small)

        small_coarse_data_grouped.append(small_coarse_data)
        big_fine_data_grouped.append(big_fine_data)

        small_coarse_share_data_grouped.append(small_coarse_share_data)
        big_fine_share_data_grouped.append(big_fine_share_data)

        small_credit_list_grouped.append(single_small_credit_list)
        big_credit_list_grouped.append(single_big_credit_list)

    '''--------------------------------------------------------------'''
    malicious = copy.deepcopy(small_coarse_vars[0])

    for epoch in tqdm(range(args.epoch), desc='The training epoch'):

        if args.malice != 0:
            small_coarse_model = models.Model(img_input, small_coarse_output)
            small_coarse_vars, small_coarse_acc = client_train_one_gra_malice(
                small_coarse_model, small_coarse_vars, small_coarse_data_list, epoch, 'small_coarse', malicious,
                args.malice, num_small
            )
            del small_coarse_model

        else:
            small_coarse_model = models.Model(img_input, small_coarse_output)
            small_coarse_vars, small_coarse_acc = client_train_one_gra(
                small_coarse_model, small_coarse_vars, small_coarse_data_list, epoch, 'small_coarse'
            )
            del small_coarse_model

        tqdm.write("Small model completed, start fine model")

        big_fine_model = models.Model(img_input, big_fine_output)
        big_fine_vars, big_fine_acc = client_train_one_gra(
            big_fine_model, big_fine_vars, big_fine_data_list, epoch, 'big_fine'
        )
        del big_fine_model
        '''-------------------------------------------------------'''
        # 分组后的模型参数
        small_coarse_vars_grouped = []
        big_fine_vars_grouped = []

        # 分组后的准确率    
        small_coarse_acc_grouped = []
        big_fine_acc_grouped = []

        # 遍历节点分组列表
        for group in nodes_blocks_index_list:
            # 模型参数
            small_coarse_var_list = []
            big_fine_var_list = []

            # 准确率
            s_acc_temp = []
            b_acc_temp = []

            for index in group:
                if index < num_small:
                    small_coarse_var_list.append(small_coarse_vars[index])
                    s_acc_temp.append(small_coarse_acc[index])
                else:
                    big_fine_var_list.append(big_fine_vars[index - num_small])
                    b_acc_temp.append(big_fine_acc[index - num_small])

            small_coarse_vars_grouped.append(small_coarse_var_list)
            big_fine_vars_grouped.append(big_fine_var_list)

            small_coarse_acc_grouped.append(s_acc_temp)
            big_fine_acc_grouped.append(b_acc_temp)

        for index in range(len(nodes_blocks_index_list)):
            small_credit_list_grouped[index] = [1 / (1 + math.exp(-math.log(x + y))) for x, y in
                                                zip(small_credit_list_grouped[index], small_coarse_acc_grouped[index])]
            big_credit_list_grouped[index] = [1 / (1 + math.exp(-math.log(x + y))) for x, y in
                                              zip(big_credit_list_grouped[index], big_fine_acc_grouped[index])]

        if epoch >= min(((args.epoch) * 4 // 5), args.epoch - 1) and (epoch % 2 == 0):
            small_vars_temp = [small_coarse_var for _ in range(args.nodes // 2)]
            small_temp = []

            stu_gra = 'small_coarse'
            tea_gra = 'big_fine'
            M = pd.read_csv(INIT_POSITION + f'/data/{args.fine_class}-{args.coarse_class}.csv', header=None)
            for group_index in range(len(nodes_blocks_index_list)):
                cur_acc = np.array(small_coarse_acc_grouped[group_index])
                data1 = pd.DataFrame(cur_acc)
                tea_output = big_fine_output
                stu_output = small_coarse_output
                tea_model = models.Model(img_input, tea_output)

                Gv = cross_relationship(M, tea_model, big_fine_vars_grouped[group_index],
                                        small_coarse_share_data_grouped[group_index], cur_acc, group_index)
                del tea_model
                s = distillate(small_coarse_vars_grouped[group_index], big_fine_vars_grouped[group_index], Gv,
                               small_coarse_share_data_grouped[group_index], stu_gra, tea_gra, small_coarse_output,
                               big_fine_output)

                small_temp.append(s)
            s_temp_flatten = [elem for sublist in small_temp for elem in sublist]

            for i, index in enumerate(s_nodes_blocks_index_list_flatten):
                small_vars_temp[index] = s_temp_flatten[i]
            small_coarse_vars = small_vars_temp

        else:
            # 将模型的参数列表扩充到和总节点数一致
            small_vars_temp = [small_coarse_var for i in range(args.nodes // 2)]
            big_vars_temp = [big_fine_var for i in range(args.nodes // 2)]

            small_temp = []
            big_temp = []

            # 对每个分组进行模型聚合操作
            for group_index in range(len(nodes_blocks_index_list)):
                s = aggregate(small_coarse_vars_grouped[group_index], small_credit_list_grouped[group_index])
                small_temp.append(s)

                b = aggregate(big_fine_vars_grouped[group_index], big_credit_list_grouped[group_index])
                big_temp.append(b)
            s_temp_flatten = [elem for sublist in small_temp for elem in sublist]

            b_temp_flatten = [elem for sublist in big_temp for elem in sublist]

            for i, index in enumerate(s_nodes_blocks_index_list_flatten):
                small_vars_temp[index] = s_temp_flatten[i]

            for i, index in enumerate(b_nodes_blocks_index_list_flatten):
                big_vars_temp[index] = b_temp_flatten[i]

            small_coarse_vars = small_vars_temp
            big_fine_vars = big_vars_temp

    print("Program over!")
