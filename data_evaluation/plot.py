import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_pickle('../kellogg.csv.pickle')

''' print(df)
fig, ax = plt.subplots()
ax.scatter(df['w_prob'], df['m_prob'])
 '''
delta_p = df['w_prob'] - df['m_prob']
plt.scatter(delta_p, df['DDG'])
plt.xlabel('∆P'), plt.ylabel('∆G')
plt.show()
