import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
from math import log

class Visualize():
    def __init__(self):
        self.datasets = [
            { 'path': 'data/result_dataframes/kellogg.pickle', 'name': 'kellogg'},
            { 'path': 'data/result_dataframes/guerois.pickle', 'name': 'guerois'},
            { 'path': 'data/result_dataframes/curatedprotherm.pickle', 'name': 'protherm'},
            { 'path': 'data/result_dataframes/potapov.pickle', 'name': 'potapov'}
        ]
        self.dataframes = self.read_datasets()

    def read_datasets(self):
        d = {}
        for data in self.datasets:
            d[data['name']] = pd.read_pickle(data['path'])
        return d

    def calculate_ddg(self):
        for key in self.dataframes.keys():
            data = self.dataframes[key]

            log = lambda x: np.log(float(x))
            dG_w = data['w_prob'].apply(log) - data['w_u_prob'].apply(log)
            dG_m = data['m_prob'].apply(log) - data['m_u_prob'].apply(log)

            self.dataframes[key]['predicted_ddg'] = ddG = dG_w - dG_m

    def create_coordinate_list(self):
        coordinate_list = []
        for dataset in self.datasets:
            df = pd.read_pickle(dataset['path'])
            delta_p = df.w_prob.apply(lambda x: np.log(float(x))) - df.m_prob.apply(lambda x: np.log(float(x)))
            coordinate_list.append(
                {'name': dataset['name'], 'x': delta_p, 'y': df['DDG']}
            )
        return coordinate_list

    def plot(self):
        coordinate_list = self.create_coordinate_list()
        for i, data in enumerate(coordinate_list):
            pcorr, pvalue  = scipy.stats.pearsonr(data['x'], data['y'])
            print(pcorr)
            plt.subplot(2, 2, 1+i)
            plt.title("Dataset: {0}, p_corr={1:.2f}".format(data['name'], pcorr))
            plt.scatter(data['x'], data['y'])
            plt.xlabel('∆logP'), plt.ylabel('∆G')

        plt.show()


visualize = Visualize()
''' visualize.plot() '''
print(visualize.dataframes)
visualize.calculate_ddg()
print(visualize.dataframes['kellogg'][['DDG', 'predicted_ddg']])
visualize.plot()
