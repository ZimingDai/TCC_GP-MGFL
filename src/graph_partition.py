import argparse
import csv
import os
import pickle

import kahypar as kahypar
import networkx as nx
import tensorflow as tf

from sklearn.cluster import SpectralClustering

from model_function import *
from prepare_csv_func import *
from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def random_split(num, total):
    # 生成num-1个随机数
    rand_nums = sorted(random.sample(range(1, total), num - 1))
    rand_nums.append(total)
    # 计算每个数值
    res = [rand_nums[i] - rand_nums[i - 1] for i in range(1, len(rand_nums))]
    res.append(total - sum(res))
    return res


def compute_edge_cost(distance, b, power, n0, datasize=100):
    # 计算边中信息传输时延以及信息传输能耗
    r = b * math.log2(1 + power / (n0 * b))
    t_transmission = datasize / r
    e = power * t_transmission

    t_delay = distance / 3

    return e + t_delay


def normalize(arr):
    """对一个一维数组进行归一化"""
    max_val = max(arr)
    min_val = min(arr)
    return [(x - min_val) / (max_val - min_val) for x in arr]


def add_arrays(arr1, arr2):
    """对两个一维数组中的每个元素进行相加"""
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result

def greedy_partition_3(G):
    # 初始化三个节点组
    group1 = []
    group2 = []
    group3 = []

    # 根据节点权重进行排序
    nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['weight'])

    # 依次将节点分配到总权重最小的组中
    for node in nodes:
        total_weight = [sum(G.nodes[n]['weight'] for n in group1),
                       sum(G.nodes[n]['weight'] for n in group2),
                       sum(G.nodes[n]['weight'] for n in group3)]
        min_weight_group = total_weight.index(min(total_weight))

        if min_weight_group == 0:
            group1.append(node)
        elif min_weight_group == 1:
            group2.append(node)
        else:
            group3.append(node)

    return [group1, group2, group3]

if __name__ == '__main__':
    clean_log()
    '''-----------超参数定义--------------'''
    parser = argparse.ArgumentParser()

    # 构建图点与边权重的超参数
    parser.add_argument("--nodes", default=30, type=int, required=True,
                        help="Total number of edge devices, 50/50 coarse and fine")

    parser.add_argument("--device", default="cpu", type=str,
                        help="Choose the device to train (now only can use cuda:0). eg: 0, default is cpu")

    parser.add_argument("--epoch", default=15, type=int,
                        help="The number of epochs required to compute the data for the partition")

    parser.add_argument("--k", default=2, required=True, type=int, help="Group number")

    parser.add_argument("--dataset", default="cifar100", type=str, help="Dataset name")

    parser.add_argument("--coarse_class", required=True, type=int,
                        help="Number of categories in a coarse-grained dataset")

    parser.add_argument("--fine_class", required=True, type=int, help="Number of categories in a fine-grained dataset")

    parser.add_argument("--imbalance", default=0.03, type=float,
                        help="The imbalance parameters of the kahypar algorithm")
    
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed")
    
    parser.add_argument("--method", default="MGGFL", type=str,
                        help="What method to cluster? eg. MGGFL, Greedy, Random, Spectral.")

    N_0 = 174
    p_max = 12
    p_min = 1
    B_total = 20000000
    channel_gain = 1
    d_min = 10
    d_max = 3000
    CPU_cycle_max = 30000
    CPU_cycle_min = 10000

    # Kahypar 需要的数据

    args = parser.parse_args()
    
    set_seed(args.seed)

    num_edges = args.nodes * (args.nodes - 1) // 2  # 构建全连通图

    if args.device != "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        tf.config.experimental_run_functions_eagerly(True)
        devices = tf.config.experimental.list_physical_devices('GPU')
        print("useable device:", devices)
        tf.config.experimental.set_memory_growth(devices[0], True)
    '''--------------------------------------------------'''

    # 图的构建（除去指导参数的引入）
    G = nx.Graph()
    # TODO:注意，现在的B，边，边的权重，节点权重都是随机的！
    B_n_list = random_split(num_edges, B_total)
    CPU_cycle_list = [random.randint(CPU_cycle_min, CPU_cycle_max) for i in range(args.nodes)]

    for i in range(args.nodes):
        if i < num_edges / 2:
            # 1-15 是小模型
            G.add_node(i, weight=CPU_cycle_list[i], type="small_coarse_model", model=None, p=None)
        else:
            # 16-30 是大模型
            G.add_node(i, weight=CPU_cycle_list[i], type="big_fine_model", model=None, p=None)

    # 生成全连通图
    for start in G.nodes():
        for end in G.nodes():
            if start < end:
                dis = random.randint(d_min, d_max)
                power = random.randint(p_min, p_max)
                b_ab = B_n_list.pop()
                weight = compute_edge_cost(dis, b_ab, power, N_0)
                G.add_edge(start, end, weight=weight, distance=dis, B=b_ab, power=power, pi=0.0)
            else:
                continue


    '''--------------------模型相互指导参数获得--------------------'''
    coarse_class = [0, 1, 2, 3, 4, 5]
    fine_class = [0, 1, 2, 3, 4, 5]

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

    if not os.path.exists(DATA_POSITION + '/' + f"{args.fine_class}-{args.coarse_class}" + '.csv'):
        print("begin createing mapping csv")
        mapp(y_fine, y_coarse, args.fine_class, args.coarse_class,
            attri=f"{args.fine_class}-{args.coarse_class}")
    
    file_path = f'../log/Gv_{args.dataset}_{args.nodes}.csv'
    if not os.path.exists(file_path):
        small_coarse_vars = [small_coarse_var for _ in range(args.nodes // 2)]
        big_fine_vars = [big_fine_var for _ in range(args.nodes // 2)]

        for partition_epoch in tqdm(range(args.epoch), desc='The training epoch'):
            small_coarse_model = models.Model(img_input, small_coarse_output)
            small_coarse_vars, small_coarse_acc = client_train_one_gra(
                small_coarse_model, small_coarse_vars, small_coarse_data_list, partition_epoch, 'small_coarse'
            )
            del small_coarse_model

            tqdm.write("Small model completed, start fine model")

            big_fine_model = models.Model(img_input, big_fine_output)
            big_fine_vars, big_fine_acc = client_train_one_gra(
                big_fine_model, big_fine_vars, big_fine_data_list, partition_epoch, 'big_fine'
            )
            del big_fine_model

            whole_acc = [small_coarse_acc, big_fine_acc]

            if partition_epoch == args.epoch - 1:
                stu_gra = 'small_coarse'
                tea_gra = 'big_fine'
                M = pd.read_csv(INIT_POSITION + '/data/' + f"{args.fine_class}-{args.coarse_class}.csv", header=None)
                cur_acc = np.array(whole_acc[0])
                data1 = pd.DataFrame(cur_acc)
                data1.to_csv(LOG_POSITION + stu_gra + '_partition_acc' + '.csv', mode='a', header=False, index=False)
                tea_output = big_fine_output
                tea_model = models.Model(img_input, tea_output)

                Gv = cross_relationship_partition(M, tea_model, big_fine_vars, small_coarse_share_list, cur_acc,
                                                args.dataset, args.nodes)

                break
        '''-----------------------------------------------------------'''

    # 读取Gv.csv文件中的二维数组
    Gv = np.loadtxt(INIT_POSITION + f'/log/Gv_{args.dataset}_{args.nodes}.csv', delimiter=',')

    # 将Gv中的每个元素设置为对应边的pi属性
    for i in range(args.nodes // 2):
        for j in range(args.nodes // 2):
            G[i][j + (args.nodes // 2)]['pi'] = Gv[i][j]

    if args.method == "MGGFL":
        print("Use MGGFL to do. ^_^")
        '''-------------------------'''
        edges = [item for sublist in G.edges() for item in sublist]
        edges = [e for e in edges]  # 对networkx形成的edge进行平铺，用于kahypar
        edge_indices = [i for i in range(0, len(edges) + 1, 2)]  # edges中哪些节点构成边

        node_weights = [int(G.nodes[i]['weight']) for i in G.nodes()]  # 每个节点的权重
        edge_weights = [int(G.edges[e]['weight']) for e in G.edges()]  # 每条边的权重
        edge_pi = [int(G.edges[e]['pi']) for e in G.edges()]  # 每条边pi的数值 
        edge_pi = np.array(edge_pi)
        '''------------------这里去控制权重控制------------------'''
        # TODO: WARNING!!
        edge_pi = 1 * edge_pi
        edge_pi = edge_pi.tolist()
        edge_weights = normalize(edge_weights)
        # edge_weights = add_arrays(edge_pi, edge_weights)

        edge_weights = [int(x * 100) for x in edge_pi]

        hypergraph = kahypar.Hypergraph(args.nodes, num_edges, edge_indices, edges, args.k, edge_weights, node_weights)

        context = kahypar.Context()

        # NOTE: 这里用绝对路径，相对路径会出错
        context.loadINIconfiguration(INIT_POSITION + "/km1_kKaHyPar_sea20.ini")

        context.setK(args.k)  # 设置组数
        context.setEpsilon(args.imbalance)  # 设置不平衡参数

        kahypar.partition(hypergraph, context)
        nodes_blocks_index_list = [[] for _ in range(args.k)]

        edges_blocks_index_list = [[] for _ in range(args.k)]

        # 获取每一个节点所在的区块ID
        for p in hypergraph.nodes():
            nodes_blocks_index_list[hypergraph.blockID(p)].append(p)

        for start in range(0, len(edges), 2):
            for i, array in enumerate(nodes_blocks_index_list):
                if edges[start] in array and edges[start + 1] in array:
                    edges_blocks_index_list[i].append((edges[start], edges[start + 1]))

        '''------------------------------------------------------'''
        # # 展示边的属性
        # for a, b, attrs in G.edges(data=True):
        #     print(f"Edge {a, b}: {attrs}")

        with open(INIT_POSITION + f'/data/node_partition_{args.dataset}_{args.nodes}_{args.k}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in nodes_blocks_index_list:
                writer.writerow([str(item) for item in row])
            f.close()

        with open(INIT_POSITION + f'/data/graph_{args.nodes}.gpickle', 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    elif args.method == "Greedy":
        print("Use Greedy to do. O.O")   

        nodes_blocks_index_list_greedy = greedy_partition_3(G)
        with open(INIT_POSITION + f'/data/node_partition_{args.dataset}_{args.nodes}_{args.k}_Greedy.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in nodes_blocks_index_list_greedy:
                writer.writerow([str(item) for item in row])
            f.close()
        
    elif args.method == "Spectral":
        print("Use Spectral to do. T_T")  
        nodes_blocks_index_list_spectral = [[] for i in range(k)] 

        adj_matrix = nx.to_numpy_array(G)
        weight_matrix = np.diag([G.nodes[node]['weight'] for node in G.nodes()])  # 构建权重矩阵
        similarity_matrix = np.dot(np.dot(weight_matrix, adj_matrix), weight_matrix)  # 计算相似度矩阵


        sc = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
        labels = sc.fit_predict(similarity_matrix)


        for i, group_num in enumerate(labels):
            nodes_blocks_index_list_spectral[group_num].append(i)
            
        with open(INIT_POSITION + f'/data/node_partition_{args.dataset}_{args.nodes}_{args.k}_Spectral.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in nodes_blocks_index_list_spectral:
                writer.writerow([str(item) for item in row])
            f.close()
    
    print("Program over!")