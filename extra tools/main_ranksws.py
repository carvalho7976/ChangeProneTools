
import pandas as pd
from matplotlib import pyplot as plt

data = {'ws': [2, 3, 4],
        'top10': [17, 20, 13]
        }

df = pd.DataFrame(data, columns=['ws', 'top10'])
print(df)
ax = df.plot.bar(x='ws')
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

df = pd.DataFrame({'top10': [17, 20, 13]}, index=[2, 3, 4])
ax = df.plot.bar()

ax.bar_label(ax.containers[0])
ax.set(xlabel='Window Size', ylabel='# in Top 10 (all datasets)')
plt.xticks(rotation='horizontal')
plt.savefig('results/rankws.pdf')
plt.show()
