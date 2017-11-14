import pandas as pd
import numpy as np
b_dir = './nutrition/'
#df = pd.read_excel(b_dir + '513_final.xlsx')
df = pd.read_csv('nutrients.csv')
df_fruit = df.loc[df['group'].str.contains(
    'Fruit') & df.name.str.contains(' raw')]
df_som = df_fruit[[col for col in df_fruit.columns if col not in [
    'name', 'group']]].fillna(0)

from sklearn.manifold import TSNE
tsne_model = TSNE(perplexity=40, n_components=2,
                  init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(np.asarray(df_som))

points = np.array(new_values)
import matplotlib.pyplot as plt
import seaborn as sns
colors = sns.mpl_palette("Dark2", len(df_fruit))

side = 100
totalDataPoints = side * side
data3d = np.random.uniform(low=0.0, high=1.0, size=(totalDataPoints, 3))

plt.figure(figsize=(15, 15))
plt.scatter(points[:, 0], points[:, 1], c=data3d)
for j, name in enumerate(df_fruit['name']):
    plt.annotate(name.split(',')[0], xy=(points[j][0], points[j][1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom', color=data3d[j])

plt.savefig('t_sne_2d.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


data2d = np.copy(points[:100])
data2d -= data2d.min(axis=0)
#data2d /= data2d.max()
from scipy.spatial.distance import cdist
data2d /= cdist(data2d, np.zeros((1, 2))).max()

from lapjv import lapjv
from numpy import random, meshgrid, linspace, dstack, sqrt, array, \
    float32, float64
size = 100
grid = dstack(meshgrid(linspace(0, 1, int(sqrt(size))),
                       linspace(0, 1, int(sqrt(size))))).reshape(-1, 2)
cost = cdist(data2d, grid, "sqeuclidean").astype(float64)
#cost *= 100000 / cost.max()
row_ind_lapjv, col_ind_lapjv, _ = lapjv(cost, verbose=True, force_doubles=True)

grid_jv = grid[col_ind_lapjv]
plt.figure(figsize=(15, 15))
names = df_fruit['name'].tolist()[:100]
for i, (name, start, end) in enumerate(zip(names, data2d, grid_jv)):
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_length=0.01, head_width=0.01, color=data3d[i])
    plt.annotate(name.split(',')[0], xy=(end[0], end[1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom', color=data3d[i])
# plt.show()
plt.savefig('t_sne_2d_lapjv.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


from minisom import MiniSom
W = np.asarray(df_som)
map_dim = 40
som = MiniSom(map_dim, map_dim, W.shape[1], sigma=1.0, random_seed=1)
# som.random_weights_init(W)
som.train_batch(W, len(W) * 150)

plt.figure(figsize=(14, 14))
for i, (t, vec) in enumerate(zip(df_fruit['name'], W)):
    winnin_position = som.winner(vec)
    plt.text(winnin_position[0],
             winnin_position[1] + np.random.rand() * .9,
             t.split(',')[0],
             color=data3d[i])

plt.xticks(range(map_dim))
plt.yticks(range(map_dim))
plt.grid()
plt.xlim([0, map_dim])
plt.ylim([0, map_dim])
# plt.show()
plt.savefig('t_som_2d.png', dpi=200)
plt.clf()
plt.cla()
plt.close()
