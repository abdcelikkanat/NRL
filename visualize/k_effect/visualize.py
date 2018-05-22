import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

sns.set_style('dark')
sns.set_context("paper")



x = np.arange(20, 140, 20)

fig, ax1 = plt.subplots(figsize=(6,6), nrows=1, ncols=1)

deepwalk_citeseer_base = [0.57236714975845404, 0.56826086956521737, 0.57605072463768137, 0.5772101449275362, 0.58124396135265688, 0.584867149758454]
deepwalk_citeseer_tne = [0.5258, 0.5257, 0.5323, 0.5375, 0.5421, 0.5432]
handle1 = ax1.plot(x, deepwalk_citeseer_tne, 'o-', x, deepwalk_citeseer_base, 'v-')
ax1.set_title("Citeseer TNE(DeepWalk)")
ax1.set_xlabel("The number of clusters")
#ax1.set_ylim(0.505, 0.545)
ax1.legend(handle1, ["Macro F1 Score", "Micro F1 Score"], loc=4)



plt.tight_layout()
#plt.show()
plt.savefig("./k_effect.eps", format='eps', bbox_inches='tight')


