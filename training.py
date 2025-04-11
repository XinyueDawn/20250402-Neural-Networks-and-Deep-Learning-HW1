import numpy as np
import pickle
import os
import tarfile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


# ------------------------- 数据加载与预处理 -------------------------
def load_CIFAR10(root):
    """加载并预处理CIFAR-10数据集"""
    # 解压数据集
    tar_path = os.path.join(root, 'cifar-10-python.tar.gz')
    if not os.path.exists(tar_path):
        from urllib.request import urlretrieve
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        urlretrieve(url, tar_path)

    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=root)

    # 加载数据
    def load_batch(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        X = data[b'data'].astype(np.float32)
        y = np.array(data[b'labels'])
        return X, y

    data_dir = os.path.join(root, 'cifar-10-batches-py')
    X_train, y_train = [], []
    for i in range(1, 6):
        X, y = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(X)
        y_train.append(y)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))

    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # 标准化处理 (基于训练集的均值和方差)
    X_train = X_train / 255.0
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8  # 防止除零

    X_train = (X_train - mean) / std
    X_val = (X_val / 255.0 - mean) / std
    X_test = (X_test / 255.0 - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------- 数据增强 -------------------------
def random_augment(X_batch):
    """对图像进行随机水平翻转和裁剪"""
    # 随机水平翻转
    flip_mask = np.random.rand(X_batch.shape[0]) < 0.5
    X_flipped = X_batch[flip_mask].reshape(-1, 3, 32, 32)[:, :, :, ::-1].reshape(-1, 3072)
    X_batch[flip_mask] = X_flipped

    # 随机裁剪（填充后裁剪）
    padded = np.pad(X_batch.reshape(-1, 3, 32, 32),
                    ((0, 0), (0, 0), (4, 4), (4, 4)),
                    mode='constant')
    crops = []
    for i in range(X_batch.shape[0]):
        x = np.random.randint(0, 8)
        y = np.random.randint(0, 8)
        crops.append(padded[i, :, x:x + 32, y:y + 32])
    return np.array(crops).reshape(-1, 3072)


# ------------------------- 神经网络模型 -------------------------
class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, num_classes, activation='relu'):
        self.params = {}
        # He初始化
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, num_classes) * np.sqrt(2.0 / hidden_size)
        self.params['b2'] = np.zeros(num_classes)
        self.activation = activation

    def forward(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # 第一层
        z1 = X.dot(W1) + b1
        if self.activation == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))
        else:
            raise ValueError("Unsupported activation")

        # 第二层
        scores = a1.dot(W2) + b2
        return scores, (X, z1, a1)

    def backward(self, X, y, scores, cache, reg):
        N = X.shape[0]
        X, z1, a1 = cache
        W1, W2 = self.params['W1'], self.params['W2']

        # Softmax梯度
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        probs[range(N), y] -= 1
        d_scores = probs / N

        # 第二层梯度（添加L2正则化）
        dW2 = a1.T.dot(d_scores) + reg * W2
        db2 = np.sum(d_scores, axis=0)

        # 第一层梯度
        da1 = d_scores.dot(W2.T)
        if self.activation == 'relu':
            dz1 = da1 * (z1 > 0)
        elif self.activation == 'sigmoid':
            dz1 = da1 * a1 * (1 - a1)

        dW1 = X.T.dot(dz1) + reg * W1
        db1 = np.sum(dz1, axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def compute_loss(self, X, y, reg):
        scores, _ = self.forward(X)
        # 稳定Softmax计算
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 交叉熵损失
        correct_logprobs = -np.log(probs[range(X.shape[0]), y] + 1e-10)
        data_loss = np.mean(correct_logprobs)

        # 修复：通过 self.params 获取权重
        W1 = self.params['W1']
        W2 = self.params['W2']
        reg_loss = 0.5 * reg * (np.sum(W1**2) + np.sum(W2**2))

        return data_loss + reg_loss

    def predict(self, X):
        scores, _ = self.forward(X)
        return np.argmax(scores, axis=1)


# ------------------------- 训练函数 -------------------------
def train(model, X_train, y_train, X_val, y_val,
          learning_rate=5e-3, reg=1e-5, num_iters=2000,
          batch_size=256, momentum=0.95, verbose=True):
    """
    优化后的训练函数，包含：
    - 动量优化
    - 余弦学习率衰减
    - 早停机制
    - 数据增强
    """
    # 初始化
    velocity = {k: np.zeros_like(v) for k, v in model.params.items()}
    best_val_acc = -1
    best_params = model.params.copy()
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    early_stop_counter = 0
    initial_lr = learning_rate

    for it in range(num_iters):
        # 数据增强小批量
        batch_indices = np.random.choice(X_train.shape[0], batch_size)
        X_batch = random_augment(X_train[batch_indices])
        y_batch = y_train[batch_indices]

        # 前向传播与损失计算
        scores, cache = model.forward(X_batch)
        loss = model.compute_loss(X_batch, y_batch, reg)
        loss_history.append(loss)

        # 反向传播
        grads = model.backward(X_batch, y_batch, scores, cache, reg)

        # 动量更新参数
        for param in model.params:
            velocity[param] = momentum * velocity[param] - learning_rate * grads[param]
            model.params[param] += velocity[param]

        # 余弦退火学习率
        learning_rate = initial_lr * 0.5 * (1 + np.cos(np.pi * it / num_iters))

        # 每个epoch评估
        if it % 100 == 0:
            train_acc = np.mean(model.predict(X_train) == y_train)
            val_acc = np.mean(model.predict(X_val) == y_val)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {k: v.copy() for k, v in model.params.items()}
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= 5:  # 早停耐心值
                    print(f"Early stopping at iteration {it}")
                    break

            if verbose and (it % 500 == 0 or it == num_iters - 1):
                print(
                    f"Iter {it}: loss={loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, lr={learning_rate:.2e}")

    # 恢复最佳参数
    model.params = best_params
    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history
    }


# ------------------------- 超参数搜索 -------------------------
def hyperparameter_search(X_train, y_train, X_val, y_val, num_combinations=20):
    """随机超参数搜索"""
    best_model = None
    best_val_acc = -1
    results = {}

    for _ in range(num_combinations):
        # 随机采样超参数
        lr = 10 ** np.random.uniform(-4, -2.5)
        reg = 10 ** np.random.uniform(-5, -2)
        hidden_size = np.random.choice([128, 256, 512])
        momentum = np.random.choice([0.85, 0.9, 0.95])

        model = ThreeLayerNet(3072, hidden_size, 10, activation='relu')
        print(f"\nTrying lr={lr:.2e}, reg={reg:.2e}, hs={hidden_size}, momentum={momentum}")

        stats = train(model, X_train, y_train, X_val, y_val,
                      learning_rate=lr, reg=reg, momentum=momentum,
                      num_iters=2000, verbose=False)

        final_val_acc = stats['val_acc_history'][-1]
        results[(lr, reg, hidden_size, momentum)] = final_val_acc

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_model = model
            print(f"New best! Val acc: {best_val_acc:.4f}")

    print("\nBest hyperparameters:")
    print(f"lr={best_model.lr:.2e}, reg={best_model.reg:.2e}, hs={best_model.hidden_size}")
    return best_model, results


# ------------------------- 主程序 -------------------------
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_CIFAR10('./')

    # 超参数搜索（耗时较长）
    # best_model, _ = hyperparameter_search(X_train, y_train, X_val, y_val)

    # 训练
    model = ThreeLayerNet(3072, 512, 10, activation='relu')
    stats = train(model, X_train, y_train, X_val, y_val,
                  learning_rate=3e-3, reg=1e-4, momentum=0.9,
                  num_iters=2000, batch_size=256)

    # 测试结果
    test_acc = np.mean(model.predict(X_test) == y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # 可视化训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(stats['loss_history'])
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(stats['val_acc_history'], label='Validation')
    plt.plot(stats['train_acc_history'], label='Train')
    plt.title("Accuracy")
    plt.legend()
    plt.show()


# ------------------------- 权重可视化 -------------------------
def visualize_weights(model, num_filters=25):
    W1 = model.params['W1']
    # 归一化权重到 [0, 1] 范围
    W1_normalized = (W1 - W1.min()) / (W1.max() - W1.min())
    plt.figure(figsize=(10, 10))
    for i in range(num_filters):
        plt.subplot(5, 5, i + 1)
        filter = W1_normalized[:, i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(filter)
        plt.axis('off')
    plt.show()



# 训练结束后调用
visualize_weights(model)

# 假设 best_model 是训练好的模型
with open('model_weights.pkl', 'wb') as f:
    pickle.dump(model.params, f)