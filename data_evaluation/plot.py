import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import numpy as np
from math import log

class Visualize():
    def __init__(self):
        self.datasets = [
            { 'path': '../kellogg.pickle', 'name': 'Kellogg'},
            { 'path': '../guerois.pickle', 'name': 'Guerois'}
        ]

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
            plt.subplot(2, 1, 1+i)
            if i == 0:
                plt.title('∆G and ∆logP correlations')
            plt.scatter(data['x'], data['y'])
            plt.xlabel('∆P'), plt.ylabel('∆G')

        plt.show()


visualize = Visualize()
visualize.plot()
