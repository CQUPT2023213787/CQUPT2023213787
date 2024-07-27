import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from scipy.sparse import coo_matrix, lil_matrix, csr_matrix

# 数据读取和预处理部分
ratings_df = pd.read_csv(r"D:\学习资料D盘专属\2024暑假\答辩\ez_douban\ratings.csv")
movies_df = pd.read_csv(r"D:\学习资料D盘专属\2024暑假\答辩\ez_douban\movies.csv")

movies_df['movieRow'] = movies_df.index
movies_df = movies_df[['movieRow', 'movieId', 'title']]
movies_df.to_csv('moviesProcessed.csv', index=False, header=True, encoding='utf-8')
print(movies_df.head())

ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
ratings_df.to_csv('ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
print(ratings_df.head())

userNo = ratings_df['userId'].max() + 1
movieNo = ratings_df['movieRow'].max() + 1

# 使用稀疏矩阵
rows = ratings_df['movieRow'].values
cols = ratings_df['userId'].values
data = ratings_df['rating'].values
rating_sparse = coo_matrix((data, (rows, cols)), shape=(movieNo, userNo)).tocsr()

# 构建稀疏矩阵标记
record = (rating_sparse > 0).astype(int)


# 归一化函数
def normalizeRatings(rating_sparse, record_sparse):
    m, n = rating_sparse.shape
    rating_mean = np.zeros((m, 1))
    rating_norm = lil_matrix(rating_sparse.shape)
    for i in range(m):
        idx = record_sparse[i, :].toarray().nonzero()[1]
        if len(idx) > 0:
            rating_mean[i] = np.mean(rating_sparse[i, idx].toarray())
            rating_norm[i, idx] = rating_sparse[i, idx] - rating_mean[i]
    return rating_norm.tocsr(), rating_mean


rating_norm, rating_mean = normalizeRatings(rating_sparse, record)
rating_mean = np.nan_to_num(rating_mean)

# 转换为稀疏张量
rating_norm_coo = rating_norm.tocoo()
indices = torch.LongTensor([rating_norm_coo.row, rating_norm_coo.col])
values = torch.FloatTensor(rating_norm_coo.data)
rating_norm_sparse = torch.sparse.FloatTensor(indices, values, torch.Size(rating_norm.shape))

record_coo = record.tocoo()
record_indices = torch.LongTensor([record_coo.row, record_coo.col])
record_values = torch.FloatTensor(record_coo.data)
record_sparse = torch.sparse.FloatTensor(record_indices, record_values, torch.Size(record.shape))

num_features = 10
X_parameters = nn.Parameter(torch.randn(movieNo, num_features) * 0.35)
Theta_parameters = nn.Parameter(torch.randn(userNo, num_features) * 0.35)


# 损失函数和优化器
def loss_function(X, Theta, rating_norm, record):
    predictions = torch.sparse.mm(rating_norm, Theta.t())
    loss = 1 / 2 * ((predictions - rating_norm.to_dense()) * record.to_dense()) ** 2
    loss += 1 / 2 * (X ** 2).sum() + 1 / 2 * (Theta ** 2).sum()
    return loss.sum()


optimizer = optim.Adam([X_parameters, Theta_parameters], lr=0.01)

# 训练循环
penalty = movieNo * userNo
for i in range(3000):
    optimizer.zero_grad()
    loss = loss_function(X_parameters, Theta_parameters, rating_norm_sparse, record_sparse)
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        with torch.no_grad():
            X_dense = X_parameters.detach().numpy()
            Theta_dense = Theta_parameters.detach().numpy()
            predicts = X_dense @ Theta_dense.T + rating_mean
            predicts_sparse = csr_matrix(predicts)
            errors = np.mean((predicts_sparse.toarray() - rating_sparse.toarray()) ** 2)
            print('step:', i, 'train loss:%.5f' % (loss.item() / penalty), 'test loss:%.5f' % errors)

# 推荐部分
user_id = input('您要向哪位用户进行推荐？请输入用户编号：')
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
idx = 0
print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))
for i in sortedResult:
    print('评分：%.2f, 电影名：%s' % (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
    idx += 1
    if idx == 20:
        break
