import numpy as np
import umap
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import check_random_state

class SpectralLibraryBuilder:
    def __init__(self, 
                 n_components=5, 
                 n_library_size=100, 
                 umap_neighbors=30, 
                 umap_min_dist=0.1, 
                 dc_percentile=2.0, 
                 random_state=42):
        """
        初始化光谱库构建器。
        
        参数:
        - n_components (int): UMAP降维后的特征维度 (c)。
        - n_library_size (int): 最终保留的光谱库样本数量 (K)。
        - umap_neighbors (int): UMAP的邻域大小参数。
        - umap_min_dist (float): UMAP的最小距离参数。
        - dc_percentile (float): 计算密度时的截断距离(dc)在所有距离中的百分位数 (推荐 1.0 - 2.0)。
        - random_state (int): 随机种子。
        """
        self.n_components = n_components
        self.n_library_size = n_library_size
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.dc_percentile = dc_percentile
        self.random_state = check_random_state(random_state)
        
        self.umap_model = None
        self.library_indices_ = None
        self.library_embeddings_ = None

    def _compute_density_and_delta(self, distances):
        """
        核心算法：计算局部密度 (rho) 和 最小高密度距离 (delta)。
        这里使用高斯核计算密度，比截断核更平滑。
        """
        N = distances.shape[0]
        
        # 1. 确定截断距离 dc
        # 取所有两两距离的第 dc_percentile 分位数作为 dc
        # 注意：squareform后的矩阵是对称的，取下三角或展平计算分位数
        all_dists = squareform(distances)
        if all_dists.size > 0:
            dc = np.percentile(all_dists, self.dc_percentile)
        else:
            dc = 1e-5 # 避免除零

        # 2. 计算局部密度 rho (Gaussian Kernel)
        # rho_i = sum(exp(-(d_ij / dc)^2))
        rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1) - 1 # 减1是为了排除自身距离0
        
        # 3. 计算 delta
        # delta_i = min(d_ij) for all j where rho_j > rho_i
        # 对于密度最大的点，delta_i = max(d_ij)
        
        # 对密度进行降序排列
        ord_rho_idx = np.argsort(rho)[::-1]
        
        delta = np.zeros(N)
        # 用于存储比当前点密度大的最近邻索引 (可选，用于聚类分配)
        # nneigh = np.zeros(N, dtype=int) 
        
        # 处理密度最大的点
        delta[ord_rho_idx[0]] = np.max(distances[ord_rho_idx[0], :])
        
        # 处理其余点
        for i in range(1, N):
            curr_idx = ord_rho_idx[i]
            # 找到所有密度比当前点大的点
            higher_density_indices = ord_rho_idx[:i]
            # 在这些点中找到距离最近的
            min_dist = np.min(distances[curr_idx, higher_density_indices])
            delta[curr_idx] = min_dist
            
        return rho, delta

    def fit(self, X, y=None):
        """
        执行构建流程。
        
        参数:
        - X: 输入数据，形状为 (N_samples, T_time_steps, C_channels)
        """
        N, T, C = X.shape
        print(f"[Builder] Input shape: N={N}, T={T}, C={C}")

        # =========================================
        # Step 1: UMAP 降维 (Spectra Embedding)
        # =========================================
        # 按照论文，将时序展平，学习光谱流形: (N*T, C) -> (N*T, c)
        X_flat = X.reshape(-1, C)
        
        # 为了速度，如果是海量数据，这里可以引入采样训练 UMAP，全量 Transform
        # 这里展示标准全量流程
        print(f"[Builder] Training UMAP on {X_flat.shape[0]} points...")
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.umap_neighbors,
            min_dist=self.umap_min_dist,
            metric='euclidean',
            random_state=self.random_state,
            n_jobs=32,  # 与 random_state 同用时n_jobs应当为1不然结果会随机
            verbose=True,
            tqdm_kwds=dict(desc="UMAP 拟合", leave=True),
        )
        X_emb_flat = self.umap_model.fit_transform(X_flat)
        
        # 还原回时序结构: (N, T, c)
        X_embedded = X_emb_flat.reshape(N, T, self.n_components)
        self.library_embeddings_ = X_embedded # 保存所有样本的嵌入
        
        print(f"[Builder] UMAP finished. Embedded shape: {X_embedded.shape}")

        # =========================================
        # Step 2: Density Peak Sampling (Library Selection)
        # =========================================
        # 为了计算样本间距离，我们需要定义“样本”的特征向量。
        # 这里将 (T, c) 展平为 (T*c) 的向量，使用欧氏距离。
        # *注*：如果对时序偏移敏感，这里应替换为 DTW 距离矩阵计算。
        X_for_clustering = X_embedded.reshape(N, -1)
        
        print("[Builder] Computing pairwise distance matrix...")
        # pdist 计算压缩距离矩阵，squareform 转为 N*N
        dist_matrix = squareform(pdist(X_for_clustering, metric='euclidean'))
        
        print("[Builder] Computing density peaks...")
        rho, delta = self._compute_density_and_delta(dist_matrix)
        
        # 计算综合得分 gamma = rho * delta
        # 只有密度高且距离其他高密度点远的点，才是好的中心 (Cluster Centers)
        gamma = rho * delta
        
        # 选择 Top-K 个点
        if self.n_library_size > N:
            print(f"[Warning] Requested library size {self.n_library_size} > N samples. Returning all indices.")
            self.library_indices_ = np.arange(N)
        else:
            # argsort 是升序，[::-1] 反转为降序，取前 K 个
            self.library_indices_ = np.argsort(gamma)[::-1][:self.n_library_size]
            
        print(f"[Builder] Library built. Selected {len(self.library_indices_)} representative samples.")
        return self

    def select_library_from_embeddings(self, X_embedded):
        """
        仅基于已有 UMAP 嵌入做 DPS 选库（不重新训练 UMAP）。
        用于在全部正/负样本嵌入上分别选库，得到正样本光谱库、负样本光谱库。
        参数:
        - X_embedded: (N, T, c) 已通过本 builder 的 umap_model 得到的嵌入。
        返回:
        - library_indices: (K,) 或 (min(K,N),) 选中的样本索引
        - library_embeddings: (K, T, c) 或 (min(K,N), T, c) 库嵌入
        """
        N, T, c = X_embedded.shape
        X_for_clustering = X_embedded.reshape(N, -1)
        dist_matrix = squareform(pdist(X_for_clustering, metric='euclidean'))
        rho, delta = self._compute_density_and_delta(dist_matrix)
        gamma = rho * delta
        K = min(self.n_library_size, N)
        if K <= 0:
            return np.array([], dtype=np.intp), X_embedded[:0]
        library_indices = np.argsort(gamma)[::-1][:K]
        library_embeddings = X_embedded[library_indices]
        return library_indices, library_embeddings

    def get_library(self, original_X=None):
        """
        获取构建好的光谱库。
        
        参数:
        - original_X (可选): 原始数据 (N, T, C)。如果提供，将返回对应的原始光谱库。
        
        返回:
        - indices: 库样本在原始数据中的索引列表 (K,)
        - embeddings: 库样本的 UMAP 嵌入 (K, T, c)
        - original_spectra: (如果提供了 original_X) 库样本的原始数据 (K, T, C)
        """
        if self.library_indices_ is None:
            raise RuntimeError("Library not built yet. Call fit() first.")
            
        indices = self.library_indices_
        embeddings = self.library_embeddings_[indices]
        
        if original_X is not None:
            original_spectra = original_X[indices]
            return indices, embeddings, original_spectra
        
        return indices, embeddings