"""
Tiny-MLP 热启动模块

使用小型神经网络预测化学平衡的初始猜测值，加速牛顿法收敛。
网络设计：
- 输入: 原子向量 (C, H, N, O, ...) + 温度 + 压力
- 输出: 对数摩尔数初始猜测 z_0 = ln(n)
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Optional, NamedTuple, List
from functools import partial


class MLPParams(NamedTuple):
    """MLP 参数结构"""
    weights: List[jnp.ndarray]
    biases: List[jnp.ndarray]


class WarmStartMLP:
    """Tiny-MLP 热启动网络
    
    轻量级 MLP 用于预测化学平衡初值。
    
    Attributes:
        hidden_dims: 隐藏层维度列表
        n_elements: 输入元素数量
        n_species: 输出物种数量
    """
    
    def __init__(
        self,
        n_elements: int = 9,
        n_species: int = 16,
        hidden_dims: Tuple[int, ...] = (32, 32)
    ):
        """初始化网络结构
        
        Args:
            n_elements: 输入元素数量 (C, H, N, O, Cl, Al, Mg, B, K)
            n_species: 输出物种数量
            hidden_dims: 隐藏层维度
        """
        self.n_elements = n_elements
        self.n_species = n_species
        self.hidden_dims = hidden_dims
        
        # 输入维度: 元素向量 + T + P (归一化)
        self.input_dim = n_elements + 2
        
    def init_params(self, key: jnp.ndarray = None) -> MLPParams:
        """初始化网络参数
        
        使用 Xavier 初始化。
        
        Args:
            key: JAX PRNG 密钥，如果为 None 则使用默认种子
            
        Returns:
            MLPParams 参数结构
        """
        if key is None:
            key = random.PRNGKey(42)
        elif not hasattr(key, 'shape') or key.shape != (2,):
            # 如果传入的不是有效的 PRNGKey，创建一个
            key = random.PRNGKey(int(key) if jnp.isscalar(key) else 42)
        
        weights = []
        biases = []
        
        dims = [self.input_dim] + list(self.hidden_dims) + [self.n_species]
        
        for i in range(len(dims) - 1):
            key, subkey = random.split(key)
            # Xavier 初始化
            scale = jnp.sqrt(2.0 / (dims[i] + dims[i+1]))
            w = random.normal(subkey, (dims[i], dims[i+1])) * scale
            b = jnp.zeros(dims[i+1])
            weights.append(w)
            biases.append(b)
        
        return MLPParams(weights=weights, biases=biases)
    
    @staticmethod
    @jax.jit
    def forward(params: MLPParams, x: jnp.ndarray) -> jnp.ndarray:
        """前向传播
        
        Args:
            params: 网络参数
            x: 输入向量 (batch, input_dim) 或 (input_dim,)
            
        Returns:
            对数摩尔数预测 z = ln(n)
        """
        h = x
        
        # 隐藏层 (ReLU 激活)
        for i, (w, b) in enumerate(zip(params.weights[:-1], params.biases[:-1])):
            h = jnp.dot(h, w) + b
            h = jax.nn.relu(h)
        
        # 输出层 (无激活，输出 log-space)
        w_out, b_out = params.weights[-1], params.biases[-1]
        z = jnp.dot(h, w_out) + b_out
        
        # 限制输出范围 [-50, 5] (对应 n ∈ [1e-22, 150])
        z = jnp.clip(z, -50.0, 5.0)
        
        return z


def predict_initial_state(
    atom_vector: jnp.ndarray,
    T: float,
    P: float,
    mlp: WarmStartMLP,
    params: MLPParams
) -> jnp.ndarray:
    """使用 MLP 预测初始状态
    
    Args:
        atom_vector: 原子向量 (n_elements,)
        T: 温度 (K)
        P: 压力 (Pa)
        mlp: WarmStartMLP 实例
        params: 网络参数
        
    Returns:
        对数摩尔数初始猜测 z_0
    """
    # 归一化输入
    # 原子向量: 除以总原子数
    atom_sum = jnp.sum(atom_vector) + 1e-10
    atom_norm = atom_vector / atom_sum
    
    # 温度归一化: (T - 2000) / 2000 (典型范围 500-5000 K)
    T_norm = (T - 2000.0) / 2000.0
    
    # 压力归一化: log10(P / 1e9) (典型范围 1-100 GPa)
    P_norm = jnp.log10(P / 1e9 + 1e-10)
    
    # 构建输入向量
    x = jnp.concatenate([atom_norm, jnp.array([T_norm, P_norm])])
    
    # 前向传播
    z = mlp.forward(params, x)
    
    return z


def create_training_data(
    n_samples: int = 1000,
    key: jnp.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """生成训练数据
    
    使用平衡求解器生成真实的平衡状态作为标签。
    
    Args:
        n_samples: 样本数量
        key: PRNG 密钥
        
    Returns:
        (inputs, targets) 元组
    """
    if key is None:
        key = random.PRNGKey(0)
    
    # 生成随机原子向量
    key, subkey = random.split(key)
    # 典型 CHNO 炸药原子比例
    atoms = random.uniform(subkey, (n_samples, 9), minval=0, maxval=1)
    atoms = atoms.at[:, 0].set(atoms[:, 0] * 3)  # C: 0-3
    atoms = atoms.at[:, 1].set(atoms[:, 1] * 6)  # H: 0-6
    atoms = atoms.at[:, 2].set(atoms[:, 2] * 6)  # N: 0-6
    atoms = atoms.at[:, 3].set(atoms[:, 3] * 6)  # O: 0-6
    atoms = atoms.at[:, 4:].set(0)  # 其他元素暂时为 0
    
    # 生成随机温度和压力
    key, subkey = random.split(key)
    T = random.uniform(subkey, (n_samples,), minval=1000, maxval=5000)
    
    key, subkey = random.split(key)
    P = random.uniform(subkey, (n_samples,), minval=1e9, maxval=50e9)  # 1-50 GPa
    
    # 简化标签: 使用启发式规则生成
    # 真实实现应使用 solve_equilibrium
    targets = jnp.zeros((n_samples, 16))
    
    # 基于元素组成估计产物
    # N2: N/2
    targets = targets.at[:, 0].set(jnp.log(atoms[:, 2] / 2 + 1e-10))
    # H2O: H/2
    targets = targets.at[:, 2].set(jnp.log(atoms[:, 1] / 2 + 1e-10))
    # CO2 or CO: 取决于氧平衡
    O_remaining = atoms[:, 3] - atoms[:, 1] / 2
    CO2_possible = jnp.minimum(atoms[:, 0], O_remaining / 2)
    targets = targets.at[:, 1].set(jnp.log(CO2_possible + 1e-10))
    
    # 归一化输入
    atom_sums = jnp.sum(atoms, axis=1, keepdims=True) + 1e-10
    atoms_norm = atoms / atom_sums
    T_norm = (T - 2000) / 2000
    P_norm = jnp.log10(P / 1e9)
    
    inputs = jnp.concatenate([atoms_norm, T_norm[:, None], P_norm[:, None]], axis=1)
    
    return inputs, targets


def train_warmstart_mlp(
    mlp: WarmStartMLP,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    key: jnp.ndarray = None
) -> Tuple[MLPParams, List[float]]:
    """训练热启动 MLP
    
    Args:
        mlp: WarmStartMLP 实例
        inputs: 输入数据 (n_samples, input_dim)
        targets: 目标数据 (n_samples, n_species)
        n_epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
        key: PRNG 密钥
        
    Returns:
        (trained_params, losses) 元组
    """
    import optax
    
    if key is None:
        key = random.PRNGKey(42)
    
    # 初始化参数
    key, subkey = random.split(key)
    params = mlp.init_params(subkey)
    
    # 优化器
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    
    # 损失函数
    def loss_fn(params, x_batch, y_batch):
        pred = jax.vmap(lambda x: mlp.forward(params, x))(x_batch)
        return jnp.mean((pred - y_batch) ** 2)
    
    # 训练步
    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    n_samples = inputs.shape[0]
    losses = []
    
    for epoch in range(n_epochs):
        # 打乱数据
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]
        
        epoch_losses = []
        for i in range(0, n_samples, batch_size):
            x_batch = inputs_shuffled[i:i+batch_size]
            y_batch = targets_shuffled[i:i+batch_size]
            
            params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
            epoch_losses.append(loss)
        
        avg_loss = jnp.mean(jnp.array(epoch_losses))
        losses.append(float(avg_loss))
    
    return params, losses
