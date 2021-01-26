import numpy as np

from sklearn import metrics
import copy
import numpy as np
import pandas as pd
import torch
import time
from LogFiles import log
from sklearn.feature_selection import SelectPercentile

import os
from functools import reduce
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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

    def Printi(self, mypath0, algo_name, TypeLearn, y_name, DataM0, res_file):

        print('Date : ' + str(DataM0.X.index[-1]))
        print('Best ' + algo_name + TypeLearn + '_' + y_name + ': (Mean : %f) using %s' % (
            self.algo.best_score_, self.algo.best_params_))
        res_file.write('Best ' + algo_name + TypeLearn + '_' + y_name + ': (Mean : %f) using %s\n' % (
            self.algo.best_score_, self.algo.best_params_))

        # res_file_Global = log(path=mypath0, nom=algo_name + '_' + TypeLearn + '_' + y_name, create=True)
        # res_file_Global.write('Best : (Mean : %f) using %s\n' % (self.algo.best_score_, self.algo.best_params_))

        try:
            means = self.algo.cv_results_['mean_test_score']
            stds = self.algo.cv_results_['std_test_score']
        except Exception as Ex:
            means = self.algo.cv_results_['mean_test_' + self.refit]
            stds = self.algo.cv_results_['std_test_' + self.refit]

        params = self.algo.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("Mean : %f / Std : %f /  with: %r" % (mean, stdev, param))
            res_file.write("Mean : %f / Std : %f /  with: %r" % (mean, stdev, param))

            param['Mean'] = mean
            param['Std'] = stdev
            BestM = pd.DataFrame(data=param, index=[DataM0.X.index[-1]])

            if os.path.exists(mypath0 + "AllFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv"):
                BestM.to_csv(mypath0 + "AllFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv", mode='a', header=False)
            else:
                BestM.to_csv(mypath0 + "AllFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv")

        res_file.write('\n')
        res_file.write('----------------------------------------------------------------------')

        #Version CSV
        BestM = pd.DataFrame(data=self.algo.best_params_, index=[DataM0.X.index[-1]])

        if os.path.exists(mypath0 + "BestFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv"):
            BestM.to_csv(mypath0 + "BestFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv", mode='a', header=False)
        else:
            BestM.to_csv(mypath0 + "BestFit_" + algo_name + '_' + TypeLearn + '_' + y_name + ".csv")


        print(f'L apprentissage de {y_name} avec le modèle {algo_name} {TypeLearn} est terminé')



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


        elapseT=[]
        for v in DataM.X_Out.columns.values.tolist():
            # OHE
            t0 = time.perf_counter()

            X0 = DataM.X_Out.copy()
            X0.loc[:, v] = np.random.choice(X0.loc[:, v], X0.shape[0])

            if algo_name == 'Neuronal':
                X0 = torch.from_numpy(X0.values).float()
                y_pred = self.best_model.predict(X0)[:, 0]
            else:
                y_pred = self.best_model.predict(X0)
                # y_pred = np.zeros(len(X0))

            ScoreBG.append(self.f_score(DataM.Y_Out, y_pred))

            elapseT.append([time.perf_counter() - t0])

        # ScoreBG_Parallel = Parallel(n_jobs=-1)(delayed(self.RandomX)(DataM, v, algo_name) for v in DataM.X_Out.columns.values.tolist())

        Normal_Scoring = np.apply_along_axis(lambda x: (x-ScoreBG[0])/np.abs(ScoreBG[0]), 0, ScoreBG[1:])

        rank = Normal_Scoring.argsort()
        Orded_variable = DataM.X_Out.columns.values[rank].tolist()

        # ll1 = np.asarray(Normal_Scoring.reshape(1, Normal_Scoring.shape[0]))
        # ll0 = self.best_model.feature_importances_.reshape(1, self.best_model.feature_importances_.shape[0])
        #
        # pd.DataFrame(data=np.transpose(np.concatenate((ll0, ll1), axis=0)), index=DataM.X_Out.columns).to_csv("SelectV.csv")

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


    def RandomX(self, DataM, v, algo_name):

        X0 = DataM.X_Out.copy()
        X0.loc[:, v] = np.random.choice(X0.loc[:, v], X0.shape[0])

        if algo_name == 'Neuronal':
            X0 = torch.from_numpy(X0.values).float()
            y_pred = self.best_model.predict(X0)[:, 0]
        else:
            y_pred = self.best_model.predict(X0)

        return self.f_score(DataM.Y_Out, y_pred)