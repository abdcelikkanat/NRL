import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

sns.set_style('dark')
sns.set_context("paper")

deepwalk = [0.5192,0.5192,0.5192,0.5192,0.5192,0.5192]
tne_max = [0.5258,0.5257,0.5323,0.5375,0.5421,0.5432]
tne_avg = [0.5203,0.5170,0.5167,0.5157,0.5209,0.5219]
tne_min = [0.5222,0.5185,0.5167,0.5169,0.5209,0.5225]

x = np.arange(20, 140, 20)

fig, ax1 = plt.subplots(figsize=(6,6), nrows=1, ncols=1)


handle1 = ax1.plot(x, deepwalk, 'o-', x, tne_max, 'o-', x, tne_avg, 'o-', x, tne_min, 'o-')
ax1.set_title("Citeseer TNE(DeepWalk)")
ax1.set_xlabel("The number of clusters")
#ax1.set_ylim(0.505, 0.545)
ax1.legend(handle1, ["DeepWalk", "TNE(DeepWalk) Max", "TNE(DeepWalk) Avg", "TNE(DeepWalk) Min"], loc=4)



plt.tight_layout()
#plt.show()
plt.savefig("./vector_combin_method.eps", format='eps', bbox_inches='tight')


