import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats


datasets = [
    { 'path': '../kellogg.csv.pickle', 'name': 'Kellogg'}
]

coordinate_list = []
for dataset in datasets:
    df = pd.read_pickle(dataset['path'])
    delta_p = df['w_prob'] - df['m_prob']
    coordinate_list.append(
        {'name': dataset['name'], 'x': delta_p, 'y': df['DDG']}
    )

print(coordinate_list)

def plot(coordinate_list):
    for df in df_list:
    pcorr, pvalue  = scipy.stats.pearsonr(*xy)
    print(pcorr)
    plt.scatter(delta_p, df['DDG'])
    plt.xlabel('∆P'), plt.ylabel('∆G')
    plt.show()


