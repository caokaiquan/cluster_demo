
#DBSCAN
#Density-Based Spatial Clustering of Applications with Noise
#核心对象是指若某个点的密度达到算法所设定的阈值则其为核心点（即r领域内点的数量不小于minpts）


# 工作流程：1.输入数据集 2.指定半径 3.指定密度阈值
#
# 伪代码：1.标记所有对象为unvisted；
#         2.随机选择一个unvisted对象p，标记p为visited
#         3.如果p的r领域至少有minpts个对象，就创建一个新簇C，并把p添加到C里面
#         4.遍历簇的每个对象，如果对象是unvisted，则标志位visted，而且如果该对象在r领域至少有minpts个对象，把这些对象添加
#         到N，如果该对象还不是任何簇的成员，则把该对象添加到C。
#         5.输出C
#         6.其他标记为噪声



####Kmeans
import pandas as pd
from sklearn.cluster import KMeans
beer = pd.read_csv('data.txt',sep = ' ')
X = beer[['calories','sodium','alcohol','cost']]
km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)

km.labels_
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')

beer.groupby('cluster').mean()
beer.groupby('cluster2').mean()

centers = beer.groupby('cluster').mean().reset_index()

import matplotlib.pyplot as plt
colors = np.array(['red','green','blue','yellow'])
plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])

plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel("Calories")
plt.ylabel("Alcohol")


pd.scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))

scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))



###scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
km = KMeans(n_clusters=3)
km.fit(X_scaled)
beer['scaled_cluster'] = km.labels_
beer.sort_values('scaled_cluster')

##聚类评估：轮廓系数
# ai为样本i到同簇其他样本的平均距离ai，ai越小说明样本i应该被聚类到该簇
# bi为样本i到其他某簇cj的所有样本的平均距离bij，
# s（i） = （b（i）- a（i））/ max{a（i），b（i）}
# s（i）接近于1表明样本i聚类合理
# s（i）接近于-1表明样本i更应该分类到另外的簇
# 若s（i）近似于0则说明样本i在两个簇的边界上

from sklearn import metrics
score_scaled = metrics.silhouette_score(X,beer.scaled_cluster)
score = metrics.silhouette_score(X,beer.cluster)
print(score_scaled,score)


scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X,labels)
    scores.append(score)

scores

#可视化结果，看看k值为多少合适
plt.plot(list(range(2,20),scores))
plt.xlabel('number of clusters')
plt.ylabel('sihouette score')



##DBSCAN实战
from sklearn.cluster import  DBSCAN
db = DBSCAN(eps = 10,min_samples=2).fit(X)
labels = db.labels_
beer['cluster_db'] = labels
beer.sort_values('cluster_db')






# #优势：1.不需要指定簇的个数
# 2.可以发现任意形状的簇
# 3.擅长找到离群点， 在sklearn中将-1标志位离群点
# 4.只需要两个参数（minpts；半径）
#
# 劣势：1.高维数据有些困难（可以通过降维解决）
# 2.参数难以解决，可能对结果影响较大
# 3.sklearn中效率较慢