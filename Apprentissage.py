import numpy as np

from sklearn import metrics
import copy
import numpy as np
import pandas as pd
import torch

import os
from functools import reduce
#import matplotlib.pyplot as plt

class Algo:

    def __init__(self, algo=None, refit=None, weight_f=None):

        if algo is None:
            self.algo = object
        else:
            self.algo = copy.copy(algo)

        self.confusion_matrix = object
        self.best_model = object
        self.f_score = object
        self.refit = refit
        self.Selected_Col = None

        if self.refit == 'neg_mean_squared_error':
            self.f_score = lambda y_true, y_pred: -1 * metrics.mean_squared_error(y_true, y_pred)

        elif self.refit == 'neg_mean_absolute_error':
            self.f_score = lambda y_true, y_pred: -1 * metrics.mean_absolute_error(y_true, y_pred)

        elif self.refit == 'f1_macro':
            self.f_score = lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='macro')

        else:
            # Cas des fonctions de score personnalisees
            # self.f_score = lambda y_true, y_pred: getattr(Algo, self.refit)(y_true, y_pred)
            if weight_f is None:
                self.f_score = getattr(Algo, self.refit)
            else:
                self.f_score = lambda y_true, y_pred: getattr(Algo, self.refit)(y_true, y_pred, weight_f)

    # @staticmethod
    # def score_ls(y_true, y_pred):
    #     if np.std(y_pred) < 0.00001:
    #         return -1
    #     else:
    #         a = 80
    #         return ((np.exp(y_pred * a) - 1)/(np.exp(y_pred * a) + 1) * y_true).sum()

    @staticmethod
    def score_ada(y_true, y_pred, weight_f):
        if np.std(y_pred) < 0.00001:
            return -1
        else:
            return (weight_f(y_pred) * y_true).sum()

    def Variable_S(self, DataM, mypath, algo_name):
        ScoreBG = list()
        # Score du jeux de test
        if algo_name == 'Neuronal':
            X_Out = torch.from_numpy(DataM.X_Out.values).float()
            y_pred = self.best_model.predict(X_Out)
            y_pred = y_pred[:, 0]
        else:
            y_pred = self.best_model.predict(DataM.X_Out)

        ScoreBG.append(self.f_score(DataM.Y_Out, y_pred))

        for v in DataM.X_Out.columns.values.tolist():
            # OHE
            X0 = DataM.X_Out.copy()
            X0.loc[:, v] = np.random.choice(X0.loc[:, v], X0.shape[0])

            if algo_name == 'Neuronal':
                X0 = torch.from_numpy(X0.values).float()
                y_pred = self.best_model.predict(X0)[:, 0]
            else:
                y_pred = self.best_model.predict(X0)

            ScoreBG.append(self.f_score(DataM.Y_Out, y_pred))

        Normal_Scoring = np.apply_along_axis(lambda x: (x-ScoreBG[0])/np.abs(ScoreBG[0]), 0, ScoreBG[1:])

        rank = Normal_Scoring.argsort()
        Orded_variable = DataM.X_Out.columns.values[rank].tolist()

        # Creation du CSV
        pd.DataFrame(data=Normal_Scoring[rank], index=Orded_variable).to_csv(mypath + "Importance_varaibles_" + algo_name + ".csv")

        # Création du graphique
        # n = 10
        # ind = np.arange(len(Normal_Scoring[rank][0:n]))
        # width = 0.20
        # fig, ax = plt.subplots(figsize=(12, 5))
        # ax.barh(ind, -Normal_Scoring[rank][0:n], color='SkyBlue')
        #
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_xlabel('Score')
        # ax.set_title('Importance des variables ' + algo_name)
        # ax.set_yticks(ind)
        # ax.invert_yaxis()
        # ax.set_yticklabels(Orded_variable[0:n])
        # # fig.autofmt_xdate(right=5)
        # ax.legend()
        # fig.tight_layout()
        #
        # plt.savefig(mypath + 'Importance_varaibles ' + algo_name + '.pdf')
        # plt.close()

        return Normal_Scoring < 0