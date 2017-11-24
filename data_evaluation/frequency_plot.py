import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
from math import log
import pickle




def plot(data_list):
    plt.close('all')
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(4, 4))
    for i, data in enumerate(data_list):

        pos = [0,0]
        if i % 2 == 1:
            pos[0] += 1
        if i > 1:
            pos[1] +=1

        print(pos)
        name = data['name']
        data = data['data']

        x = list(data['frequencies'].keys())
        y = list(data['frequencies'].values())

        ax[tuple(pos)].bar(list(range(20)),y)
        ax[tuple(pos)].set_title(name)
        plt.xticks(list(range(20)), x)

    plt.tight_layout()
    plt.show()



''' fig.text(0.5, 0.04, 'common X', ha='center')
fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
 '''
load_data = lambda x: pickle.load(open(x, "rb"))
data_list = [
    {"name":"guerois", "data": load_data("../data/frequencies/guerois_freq.p")},
    {"name":"kellogg", "data": load_data("../data/frequencies/kellogg_freq.p")},
    {"name":"curatedprotherm", "data": load_data("../data/frequencies/curatedprotherm_freq.p")},
    {"name":"potapov", "data": load_data("../data/frequencies/potapov_freq.p")},
]

plot(data_list)
