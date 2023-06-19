#%%
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import pickle as pk
path = 'd:/Facultad/Tesis/'

phr_emb = pk.load(open(path+'phr_embbedings/phrases_lkvec_emb.pickle', 'rb'))
#matrix = df.ada_embedding.apply(eval).to_list()

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(phr_emb)
plt.figure()
plt.scatter(vis_dims[:, 0], vis_dims[:, 1])
plt.show()

# colors = ["red", "darkorange", "gold", "turquiose", "darkgreen"]
# x = [x for x,y in vis_dims]
# y = [y for x,y in vis_dims]
# color_indices = df.Score.values - 1

# colormap = matplotlib.colors.ListedColormap(colors)
# plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
# plt.title("Amazon ratings visualized in language using t-SNE")