# CIFAR-10图像分类项目

### 安装依赖
```bash
pip install -r requirements.txt
```

### 准备数据
下载数据集到 `data/` 目录：
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
```


### 参数配置
| 参数          | 默认值    | 说明                     |
|---------------|-----------|--------------------------|
| `--hidden_size` | 512      | 隐藏层神经元数量         |
| `--learning_rate` | 3e-3   | 初始学习率               |
| `--reg`         | 1e-4     | L2正则化强度             |
| `--num_iters`   | 2000     | 训练迭代次数             |

### 性能指标
| 数据集    | 准确率   |
|-----------|----------|
| 训练集    | 62.3%    |
| 验证集    | 54.8%    | 
| 测试集    | 53.1%    |
