import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

sns.set_style('dark')
sns.set_context("paper")



x = np.arange(0.1, 1.0, 0.1)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(12,6), nrows=2, ncols=3)

deepwalk_citeseer_base = [0.4588,0.4898,0.5041,0.5124,0.5161,0.5193,0.5216,0.5209,0.5234]
deepwalk_citeseer_tne = [0.4494,0.4962,0.5154,0.5297,0.5375,0.5453,0.5481,0.5559,0.5580]
handle1 = ax1.plot(x, deepwalk_citeseer_tne, 'o-', x, deepwalk_citeseer_base, 'v-')
ax1.set_title("Citeseer")
ax1.set_ylabel("Macro F1 Score")
ax1.legend(handle1, ["TNE(DeepWalk)", "DeepWalk"], loc=4)

ax2.set_title("Dblp")
deepwalk_dblp_base = [0.5463,0.5522,0.5549,0.5556,0.5557,0.5560, 0.5565, 0.5568,0.5571]
deepwalk_dblp_tne = [0.5477,0.5595,0.5646,0.5663,0.5668,0.5676,0.5668,0.5673,0.5695]
handle2 = ax2.plot(x, deepwalk_dblp_tne, 'o-', x, deepwalk_dblp_base, 'v-')
ax2.legend(handle2, ["TNE(DeepWalk)", "DeepWalk"], loc=4)

ax3.set_title("Blogcatalog")
deepwalk_blogcatalog_base = [0.1847,0.2138,0.2307,0.2404,0.2471,0.2523,0.2563,0.2598,0.2614]
deepwalk_blogcatalog_tne = [0.1644,0.1936,0.2112,0.2222,0.2309,0.2377,0.2424,0.2444,0.2472]
handle3 = ax3.plot(x, deepwalk_blogcatalog_tne, 'o-', x, deepwalk_blogcatalog_base, 'v-')
ax3.legend(handle3, ["TNE(DeepWalk)", "DeepWalk"], loc=4)





node2vec_citeseer_base = [0.4625,0.4978,0.5148,0.5240,0.5312,0.5361,0.5355,0.5368,0.5390]
node2vec_citeseer_tne = [0.4445,0.4964,0.5210,0.5390,0.5510,0.5583,0.5651,0.5716,0.5723]
handle4 = ax4.plot(x, node2vec_citeseer_tne, 'o-', x, node2vec_citeseer_base, 'v-')
ax4.set_ylabel("Macro F1 Score")
ax4.legend(handle4, ["TNE(Node2Vec)", "Node2Vec"], loc=4)

node2vec_dblp_base = [0.5488,0.5565,0.5588,0.5597,0.5597,0.5600,0.5599,0.5595,0.5615]
node2vec_dblp_tne = [0.5489,0.5621,0.5661,0.5687,0.5699,0.5707,0.5718,0.5711,0.5718]
handle5 = ax5.plot(x, node2vec_dblp_tne, 'o-', x, node2vec_dblp_base, 'v-')
ax5.legend(handle5, ["TNE(Node2Vec)", "Node2Vec"], loc=4)

node2vec_blogcatalog_base = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
node2vec_blogcatalog_tne = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
handle6 = ax6.plot(x, node2vec_blogcatalog_tne, 'o-', x, node2vec_blogcatalog_base, 'v-')
ax6.legend(handle6, ["TNE(Node2Vec)", "Node2Vec"], loc=4)




plt.tight_layout()
#plt.show()
plt.savefig("./macro_scores.eps", format='eps', bbox_inches='tight')


