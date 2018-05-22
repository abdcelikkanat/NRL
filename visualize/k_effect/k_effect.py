import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

sns.set_context("paper")

x = np.arange(100)
y = x**2

plt.subplot(2, 1, 1)
plt.plot(x, y, '-', x, 2*y, '-')
plt.title("First title")
plt.xlabel("1st x label")
plt.ylabel("1st y label")

plt.subplot(2, 1, 2)
plt.plot(x, y, 'r-')
plt.title("Second")
plt.xlabel("2nd x label")
plt.ylabel("2nd y label")

#plt.show()
plt.savefig("./s.png", bbox_inches='tight')

#g = sns.lmplot(x="xlabel", y="ylabel", truncate=True)

#g.set_axis_label("Xlabel", "Ylabel")
