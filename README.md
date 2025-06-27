# README



# Multi-Granularity Federated Learning by Graph-Partitioning

[![Paper](https://img.shields.io/badge/Paper-IEEE_TCC-blue.svg?style=for-the-badge&logo=ieee)](https://ieeexplore.ieee.org/abstract/document/10748407)

This repository contains the official implementation of the paper:

**Multi-Granularity Federated Learning by Graph-Partitioning**

> In edge computing, energy-limited and heterogeneous edge clients face challenges in communication efficiency, model performance, and security. Traditional blockchain-based federated learning (BFL) methods often fall short in addressing these issues simultaneously. We propose **GP-MGFL**, a graph-partitioning multi-granularity federated learning framework on a consortium blockchain. By using balanced graph partitioning with observer and consensus nodes, GP-MGFL reduces communication costs while maintaining effective intra-group guidance. A cross-granularity guidance mechanism enables fine-grained models to guide coarse-grained ones, enhancing overall performance. A dynamic credit-based aggregation scheme further improves adaptability and robustness. Experimental results show that GP-MGFL outperforms standard BFL by up to 5.6%, and achieves up to 11.1% improvement in malicious scenarios.

This codebase is intended for research and reproducibility purposes only.

------

## 💻 Installation

```bash
pip install -r requirements.txt
cd src/
```

Python version: **3.8+** is recommended.

------

## 🚀 How to Run

### Step 1: Graph Partitioning

Run client grouping and graph partitioning:

```bash
python graph_partition.py [arguments]
```

**Arguments:**

- `--nodes`: Number of total edge clients (half coarse, half fine). Default is `30`.
- `--device`: Device index or `cpu`. Default is `cpu`.
- `--epoch`: Number of training epochs used for partition computation. Default is `15`.
- `--k`: Number of groups. **Required**.
- `--dataset`: Dataset name (`mnist` or `cifar100`). Default is `cifar100`.
- `--coarse_class`: Number of coarse classes. **Required**.
- `--fine_class`: Number of fine classes. **Required**.
- `--imbalance`: Imbalance factor for the Kahypar partitioner. Default is `0.03`.
- `--seed`: Random seed. Default is `0`.
- `--method`: Grouping method: `MGGFL`, `Greedy`, `Random`, `Spectral`. Default is `MGGFL`.

**Example:**

```bash
python graph_partition.py --nodes 30 --device 0 --epoch 5 --k 3 \
--dataset mnist --coarse_class 6 --fine_class 10 --method Greedy
```

------

### Step 2: Training with Guidance

Run the federated training with cross-granularity guidance:

```bash
python cal_acc.py [arguments]
```

**Arguments:**

- `--nodes`: Number of total edge clients. Default is `30`.
- `--malice`: Number of malicious clients. Default is `0`.
- `--device`: Device index or `cpu`. Default is `cpu`.
- `--epoch`: Number of training epochs.
- `--k`: Number of groups. Default is `2`.
- `--dataset`: Dataset name (`mnist` or `cifar100`). Default is `cifar100`.
- `--coarse_class`: Number of coarse classes. **Required**.
- `--fine_class`: Number of fine classes. **Required**.
- `--seed`: Random seed. Default is `0`.
- `--method`: Grouping method: `MGGFL`, `Greedy`, `Random`, `Spectral`. Default is `MGGFL`.

**Example:**

```bash
python cal_acc.py --nodes 30 --device 1 --epoch 10 --k 3 \
--dataset mnist --coarse_class 6 --fine_class 10 --method Greedy
```

---



## 📁 Data and Logs Description

#### `/data`

This folder stores all necessary data and intermediate files used in the project:

- **Client partition files**: e.g., `node_partition_mnist`
- **Class mapping files**:
  - `100-20` for CIFAR-100
  - `10-6` for MNIST
- **Intermediate graph file**:
  - `graph.gpickle` for partitioning logic

#### `/log`

This folder records training outputs and intermediate matrices:

- `scale_granularityX.txt`:
  - Accuracy of each client per epoch
- `Gv*.csv`:
  - Client guidance matrices
  - `Gv20.csv`, `Gv30.csv`: for CIFAR-100 (20/30 clients)
  - `Gv_mnist_30.csv`: for MNIST with 30 clients

## 📦 Dataset

Supported datasets:

- **MNIST**
- **CIFAR-100**

Datasets are automatically downloaded and processed during execution.

------

## 📜 License

This code is released **for research purposes only**.

------

## 📖 Citation

If you find this work useful in your research, please cite the following paper:

```bibtex
@article{dai2024multi,
  title={Multi-Granularity Federated Learning by Graph-Partitioning},
  author={Dai, Ziming and Zhao, Yunfeng and Qiu, Chao and Wang, Xiaofei and Yao, Haipeng and Niyato, Dusit},
  journal={IEEE Transactions on Cloud Computing},
  year={2024},
  publisher={IEEE}
}
```
