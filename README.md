# README



### 文件结构

#### '/data'

这个文件夹用于存放项目所需要的数据文件：

1. 分组文件（`node_partition_mnist`等）
2. 类别对应文件（`100-20`为CIFAR100对应文件，`10-6`为MNIST对应文件）
3. 中间生成的图文件（`graph.gpickle`）

#### '/log'

该文件夹记录所有客户端的每个epoch的准确度，记录为`scale_granularityX.txt`，以及中间过程生成的指导能力矩阵`Gv.csv`

`Gv20(30).csv`表示CIFAR100数据集中20、30个客户端的指导矩阵；`Gv_mnist_30.csv`为MNIST数据集中30个客户端的指导矩阵

#### '/src'

包含项目的所有源代码文件。

* `data_prepare_func.py`：存放了对数据集进行处理的函数
* `model.py`：定义了大规模模型与小规模模型的结构
* `model_function.py`：定义了模型进行操作的函数，训练、蒸馏等
* `prepare_csv_func.py`：定义了生成粒度对应关系的函数
* `graph_partition.py`：进行图分割部分等分组方式的代码，对应文章Fig.5 的A - B
* `cal_acc.py`：进行模型训练与指导的代码，对应Fig.5的C - E
* `draw.py`：简单定义了模型结果画图的函数

#### '/test'

纯粹测试，无意义

### 运行方式

#### 前提条件

```sh
pip3 install -r requirements.txt
cd /src
```

#### 分组

使用以下命令行参数来运行程序：

```sh
python graph_patition.py --nodes <节点数> --device <设备类型> --epoch <训练轮数> --k <组数> --dataset <数据集名> --coarse_class <粗分类别数> --fine_class <细分类别数> --imbalance <不平衡参数> --seed <随机种子> --method <聚类方法>
```

**参数说明：**

- `--nodes`: 边缘设备的总数目，一半粗一半细。默认为 30。
- `--device`: 选择训练设备，例如：0，默认为 cpu。
- `--epoch`: 计算数据分区所需的训练轮数。默认为 15。
- `--k`: 组数。必须指定。
- `--dataset`: 使用的数据集名称，默认为 "cifar100"。
- `--coarse_class`: 粗粒度数据集中的分类数量。必须指定。
- `--fine_class`: 细粒度数据集中的分类数量。必须指定。
- `--imbalance`: kahypar 算法的不平衡参数。默认为 0.03。
- `--seed`: 随机种子，默认为 0。
- `--method`: 聚类方法选择，例如：MGGFL, Greedy, Random, Spectral。默认为 "MGGFL"。

**示例：**

```sh
python graph_partition.py --nodes 30 --device 0 --epoch 5 --k 3 --dataset mnist --coarse_class 6 --fine_class 10 --method Greedy
```



#### 训练

要运行程序，请使用以下命令格式，并根据需要替换各参数的值：

```sh
python cal_acc.py --nodes <节点数> --malice <恶意节点数> --device <设备类型> --epoch <训练轮数> --k <组数> --dataset <数据集名称> --coarse_class <粗分类别数> --fine_class <细分类别数> --seed <随机种子> --method <聚类方法>
```

**参数说明：**

- `--nodes`: 边缘设备的总数目。默认为 30。
- `--malice`: 网络中恶意节点的数量。默认为 0。
- `--device`: 训练设备的选择，例如：0，默认为 cpu。
- `--epoch`: 执行的训练轮数。
- `--k`: 分组数量。默认为 2。
- `--dataset`: 使用的数据集名称，默认为 "cifar100"。
- `--coarse_class`: 粗粒度数据集中的分类数量。必须指定。
- `--fine_class`: 细粒度数据集中的分类数量。必须指定。
- `--seed`: 随机种子。默认为 0。
- `--method`: 选择的聚类方法，例如：MGGFL, Greedy, Random, Spectral。默认为 "MGGFL"。

**示例：**

```sh
python cal_acc.py --nodes 30 --device 1 --epoch 10 --k 3 --dataset mnist --coarse_class 6 --fine_class 10 --method Greedy
```

