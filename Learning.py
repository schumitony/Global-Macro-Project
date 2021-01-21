from Apprentissage import Algo

from xgboost import XGBRegressor, XGBClassifier
from functools import partial

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import make_scorer

from skorch import NeuralNet

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import os
import pickle
from NeuronalNetwork import Net
import csv
from functools import reduce
import copy
# import matplotlib.pyplot as plt
import re

import pathlib

import torch.optim as optim
import torch.nn as nn
import torch
import skorch as sk


class Learnings:

    def __init__(self, learns=None, DataM=None, list_model=None, list_refit=None,
                 kfold=None, mypath=None, mypath0=None, weight_f=None,
                 save_path=None, load_path=None, stratification=None):

        self.algo_dict = dict()
        self.best_score_cv = dict()
        self.best_score_out = dict()
        self.Selected_Col = None

        if learns is None:

            if isinstance(list_model, list):
                for m, refit in zip(list_model, list_refit):
                    self.algo_dict.update({m: Algo(refit=refit, weight_f=weight_f)})
                    self.best_score_cv.update({m: None})
                    self.best_score_out.update({m: None})
            else:
                self.algo_dict.update({list_model: Algo(refit=list_refit, weight_f=weight_f)})
                self.best_score_cv.update({list_model: None})
                self.best_score_out.update({list_model: None})

            self.best_model_ever = object
            self.kfold = kfold
            self.DataM = DataM
            self.weight_f = weight_f
            self.mypath = mypath
            self.mypath0 = mypath0
            self.save_path = save_path
            self.load_path = load_path


        else:
            # Lorsque l'on réapprend un modèle apres une selection de variable, on copie le précedent
            self.best_model_ever = object
            self.kfold = copy.copy(learns.kfold)
            # self.strat = copy.copy(learns.strat)
            self.DataM = copy.copy(learns.DataM)
            self.mypath = learns.mypath
            self.mypath0 = learns.mypath0
            self.save_path = learns.save_path
            self.load_path = learns.load_path

            for m, a in learns.algo_dict.items():
                self.algo_dict.update({m: Algo(a.algo, a.refit, learns.weight_f)})
                self.best_score_cv.update({m: None})
                self.best_score_out.update({m: None})

                # if self.strat is not None:
                #     generator = self.TrainTestGenerator()
                # else:
                #     generator = self.kfold.n_splits
                # self.algo_dict[m].algo.cv = generator

    def load_model(self):

        for m in self.algo_dict.keys():

            if m == 'Neuronal':

                scoring = ["f1_macro"]

                p_grid = {"batch_size": [120], "lr": [0.1], "max_epochs": [500]}
                # p_grid = {"batch_size": [130, 260], "lr": [0.01, 0.05, 0.1], "max_epochs": [200]}

                model = NeuralNet(
                    module=Net,
                    criterion=nn.MSELoss,
                    # max_epochs=20,
                    # batch_size=128,
                    # lr=0.1,
                    optimizer=optim.SGD,
                    # optimizer__momentum=0.5,
                    module__Imput_dim=100,
                    callbacks=[sk.callbacks.EarlyStopping()]
                )

            if m == 'RandForest':
                # RandomForestClassifier
                scoring = ["f1_macro"]
                p_grid = {"max_features": [0.2], "max_depth": [20, 50], "n_estimators": [500, 650]}
                # p_grid = {"max_features": [0.2], "max_depth": [5], "n_estimators": [100]}
                model = RandomForestClassifier()

            if m == 'XGBClassifier':
                # XGBClassifier
                scoring = ["f1_macro"]
                p_grid = {"max_depth": [20, 50], "n_estimators": [500, 650]}
                # p_grid = {"max_depth": [5], "n_estimators": [100]}
                model = XGBClassifier(learning_rate=0.1)

            if m == 'AdaB':
                # AdaBoostClassifier
                scoring = ["f1_macro"]
                p_grid = {"learning_rate": [0.025, 0.05, 0.1, 0.15], "n_estimators": [200, 350, 500, 650]}
                model = AdaBoostClassifier()

            if m == 'GBoostM':
                # GradientBoostingClassifier
                scoring = ["f1_macro"]
                p_grid = {"learning_rate": [0.025, 0.05, 0.1, 0.15], "max_depth": [10, 20, 30], "max_features": [0.2, 0.4]}
                model = GradientBoostingClassifier(n_estimators=350)

            if m == 'DecisionTreeClassifier':
                # DecisionTreeClassifier
                scoring = ["f1_macro"]
                p_grid = {"min_samples_leaf": [0.01, 0.025, 0.05], "splitter": ["best", "random"]}
                model = DecisionTreeClassifier()

            if m == 'RadiusNeigh':
                # KNeighborsClassifier
                scoring = ["f1_macro"]
                p_grid = {"weights": ['uniform', 'distance'], "n_neighbors": list(range(1001, 10001, 1000))}
                # p_grid = {"weights": ['uniform'], "n_neighbors": [1001]}
                model = KNeighborsClassifier(algorithm='auto')

            if m == 'SVC':
                # SVC
                scoring = ["f1_macro"]
                p_grid = {'kernel': ['rbf', 'sigmoid'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
                model = SVC()

            if m == 'Ridge':
                # Ridge
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {"alpha": list(range(300, 4001, 100))}
                model = Ridge(fit_intercept=True)

            if m == 'XGBRegressor':
                # XGBRegressor
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {"learning_rate": [0.05, 0.075, 0.1], "max_depth": [5, 20, 100, 200, 400, 800], "n_estimators": [100, 200, 300, 400, 500, 600]}
                # p_grid = {"learning_rate": [0.1], "max_depth": [5], "n_estimators": [50, 100]}
                model = XGBRegressor()

            if m == 'RandForestReg':
                # RandomForestRegressor
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {"max_features": [0.05, 0.1, 0.2], "max_depth": [5, 20, 100, 200], "n_estimators": [100, 200, 300, 400]}
                #p_grid = {"max_features": [0.2], "max_depth": [5], "n_estimators": [50, 60]}
                model = RandomForestRegressor()

            if m == 'AdaBReg':
                # AdaBoostRegressor
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_gri2d = {"learning_rate": [0.05, 0.075, 0.1], "n_estimators": [100, 200, 300, 400]}
                # p_grid = {"learning_rate": [0.01], "n_estimators": [50, 60]}
                model = AdaBoostRegressor()

            if m == 'DecisionTreeRegressor':
                # DecisionTreeRegressor
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {"min_samples_leaf": [0.01, 0.025, 0.05], "splitter": ["best", "random"]}
                model = DecisionTreeRegressor()

            if m == 'SVR':
                # SVR
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {'kernel': ['rbf', 'sigmoid'], 'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
                model = SVR()

            if m == 'GBoostMReg':
                # GradientBoostingRegressor
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
                p_grid = {"learning_rate": [0.025, 0.05], "max_depth": [20, 100], "max_features": [0.2, 0.4]}
                model = GradientBoostingRegressor(n_estimators=450)

            if 'score_' in self.algo_dict[m].refit:
                if self.weight_f is None:
                    scoring = make_scorer(getattr(Algo, self.algo_dict[m].refit), greater_is_better=True)
                else:
                    scoring = make_scorer(getattr(Algo, self.algo_dict[m].refit), weight_f=self.weight_f,
                                          greater_is_better=True)

                refit = True
            else:
                refit = self.algo_dict[m].refit

            self.algo_dict[m].algo = GridSearchCV(estimator=model, param_grid=p_grid, cv=self.kfold, n_jobs=-1,
                                                  scoring=scoring, refit=refit, verbose=2)

    def Learn_Algo(self, res_file, TypeLearn="", load_mode=False):

        y_name = self.DataM.Y.name

        for algo_name, a in self.algo_dict.items():

            #Apprentissage du modèle
            DataM0 = self.DataM.SelectData(a.Selected_Col)

            self.x = print(' ')
            print('Debut de l''apprentissage ' + algo_name + TypeLearn + '_' + y_name)

            save_file = self.save_path + 'Learn_' + y_name + '_' + algo_name + TypeLearn + '.pkl'
            load_file = self.load_path + 'Learn_' + y_name + '_' + algo_name + TypeLearn + '.pkl'

            if hasattr(a.algo.estimator, 'module__Imput_dim'):
                a.algo.estimator.module__Imput_dim = DataM0.X.shape[1]

                DataM0.X = torch.from_numpy(DataM0.X.values).float()
                DataM0.X_Out = torch.from_numpy(DataM0.X_Out.values).float()

                DataM0.Y = DataM0.Y.values
                DataM0.Y_Out = DataM0.Y_Out.values

            if load_mode and os.path.exists(self.load_path) and os.path.isfile(load_file):
                a.algo = pickle.load(open(load_file, 'rb'))

            else:
                a.algo.fit(DataM0.X, DataM0.Y)

                # Sauvgarde de l'object self avec pickle
                if not os.path.exists(self.save_path):
                    pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

                with open(save_file, 'wb') as output:
                    pickle.dump(a.algo, output, pickle.HIGHEST_PROTOCOL)

            # if load_mode is False:
            #     a.algo.fit(DataM0.X, DataM0.Y)
            #     # Sauvgarde de l'object self avec pickle
            #     the_path = self.mypath0 + 'Pickle\\WorkInProgress\\'
            #     if not os.path.exists(the_path):
            #         pathlib.Path(the_path).mkdir(parents=True, exist_ok=True)
            #
            #     with open(the_path + 'Learn_' + y_name + '_' + algo_name + TypeLearn + '.pkl', 'wb') as output:
            #          pickle.dump(a.algo, output, pickle.HIGHEST_PROTOCOL)
            # else:
            #
            #     the_path = self.mypath0 + "Pickle\\" + 'Learn_' + y_name + '_' + algo_name + TypeLearn + '.pkl'
            #     if os.path.isfile(the_path):
            #         a.algo = pickle.load(open(the_path, 'rb'))
            #     else:
            #         a.algo.fit(DataM0.X, DataM0.Y)
            #         # Sauvgarde de l'object self avec pickle
            #         if not os.path.exists(self.mypath0 + 'Pickle\\'):
            #             pathlib.Path(self.mypath0 + 'Pickle\\').mkdir(parents=True, exist_ok=True)
            #
            #         with open(self.mypath0 + "Pickle\\" + 'Learn_' + y_name + '_' + algo_name + TypeLearn + '.pkl',
            #                   'wb') as output:
            #             pickle.dump(a.algo, output, pickle.HIGHEST_PROTOCOL)

            # Modèle calé
            a.best_model = a.algo.best_estimator_

            self.best_score_cv[algo_name] = a.algo.best_score_

            # Score du modèle sur l'echantillon hors apprentissage
            self.best_score_out[algo_name] = a.f_score(DataM0.Y_Out, a.best_model.predict(DataM0.X_Out))
            if a.refit == "f1_macro":
                a.confusion_matrix = confusion_matrix(DataM0.Y_Out, a.best_model.predict(DataM0.X_Out))

            print('Best ' + algo_name + TypeLearn + '_' + y_name + ': (Mean : %f) using %s' % (a.algo.best_score_, a.algo.best_params_))
            res_file.write('Best ' + algo_name + TypeLearn + '_' + y_name + ': (Mean : %f) using %s\n' % (a.algo.best_score_, a.algo.best_params_))

            try:
                means = a.algo.cv_results_['mean_test_score']
                stds = a.algo.cv_results_['std_test_score']
            except Exception as Ex:
                means = a.algo.cv_results_['mean_test_' + a.refit]
                stds = a.algo.cv_results_['std_test_' + a.refit]

            params = a.algo.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("Mean : %f / Std : %f /  with: %r" % (mean, stdev, param))
                res_file.write("Mean : %f / Std : %f /  with: %r" % (mean, stdev, param))

            res_file.write('\n')
            res_file.write('----------------------------------------------------------------------')

            print(f'L apprentissage de {y_name} avec le modèle {algo_name} {TypeLearn} est terminé')

        # Selection du meilleur modele entre les différents algo (sur la base du score des prediction out of sample)
        v = list(self.best_score_out.values())
        k = list(self.best_score_out.keys())
        vM = k[v.index(max(v))]
        self.best_model_ever = self.algo_dict[vM].best_model
        self.Selected_Col = self.algo_dict[vM].Selected_Col

    def Predict_CoutMoy2(self, y_test, Mu):

        y_name = self.DataM.Y.name

        for m, a in self.algo_dict.items():
            if a.Selected_Col is not None:
                cc = "_Selected"
            else:
                cc = ""

            DataM0 = self.DataM.OHE_selected(a.Selected_Col)
            # Prediction de notre jeu de Test
            y_test[y_name + '_' + m + cc] = self.algo_dict[m].best_model.predict(DataM0.X_test_OHE)

            # Creation de la série des predictions indexé par le num de police
            y_test[y_name + '_' + m + cc] = pd.Series(y_test[y_name + '_' + m + cc], index=DataM0.y_PolNum)
            y_test[y_name + '_' + m + cc] = y_test[y_name + '_' + m + cc].apply(lambda x: x*Mu)


        # Prediction de notre jeu de Test
        DataM0 = self.DataM.OHE_selected(self.Selected_Col)
        y_test[y_name + cc] = self.best_model_ever.predict(DataM0.X_test_OHE)

        # Creation de la série des predictions indexé par le num de police
        y_test[y_name + cc] = pd.Series(y_test[y_name + cc], index=DataM0.y_PolNum)
        y_test[y_name + cc] = y_test[y_name + cc].apply(lambda x: x * Mu)




    def BlendModel(self):

        y_name = self.DataM.Y.name
        y_blend = dict()
        ll = list()

        for m, a in self.algo_dict.items():
            # Prediction de notre jeu Out of Train
            y_blend[y_name + '_Out_' + m] = self.algo_dict[m].best_model.predict(self.DataM.X_Out)

            # Creation de la série des predictions indexé par le num de police
            y_blend[y_name + '_Out_' + m] = pd.Series(y_blend[y_name + '_Out_' + m], index=self.DataM.Y_Out.index)
            y_blend[y_name + '_Out_' + m].name = y_name + '_Out_' + m

            ll.append(y_blend[y_name + '_Out_' + m])

        Y_predic = pd.DataFrame(np.stack(ll, axis=-1))
        ll.append(self.DataM.Y_Out)

        pd.DataFrame(np.stack(ll, axis=-1)).to_csv(self.mypath + "\\Predict.csv")

        # Predicteur melangé par Regression lineaire
        scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
        p_grid = {"alpha": list(np.arange(0.01, 10, 0.1))}
        model = Ridge(fit_intercept=True)

        A = GridSearchCV(estimator=model, param_grid=p_grid, n_jobs=-1,
                         scoring=scoring, refit='neg_mean_squared_error', verbose=2)

        A.fit(Y_predic, self.DataM.Y_Out)


    def Predict(self, y_pred, BestAlgo=None):

        y_name = self.DataM.Y.name
        #ll = list()

        for m, a in self.algo_dict.items():
            DataM0 = self.DataM.SelectData(a.Selected_Col)

            if m == 'Neuronal':
                DataM0.X_BT = torch.from_numpy(DataM0.X_BT.values).float()

            if a.Selected_Col is not None:
                cc = "_Selected"
            else:
                cc = ""

            # Prediction de notre jeu de BackTest
            y_pred[y_name + '_' + m + cc] = self.algo_dict[m].best_model.predict(DataM0.X_BT)

            if y_pred[y_name + '_' + m + cc].ndim == 2:
                y_pred[y_name + '_' + m + cc] = y_pred[y_name + '_' + m + cc][:, 0]

            # Creation de la série des predictions indexé par le num de police
            y_pred[y_name + '_' + m + cc] = pd.Series(y_pred[y_name + '_' + m + cc], index=DataM0.Y_BT.index)
            y_pred[y_name + '_' + m + cc].name = y_name + '_' + m + cc

            #ll.append(y_pred[y_name + '_' + m + cc])

        #ll.append(DataM0.Y_BT)
        #df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), list(map(lambda x: x.to_frame(), ll)))
        #pd.DataFrame(df).to_csv(self.mypath + "Predict_" + y_name + cc + ".csv")

        # Prediction du meilleur modèle
        if BestAlgo is not None:
            DataM0 = self.DataM.SelectData(self.Selected_Col)
            y_pred[y_name + "_BestAlgo" + cc] = self.best_model_ever.predict(DataM0.X_BT)

            # Creation de la série des predictions indexé par les dates
            y_pred[y_name + "_BestAlgo" + cc] = pd.Series(y_pred[y_name + "_BestAlgo" + cc], index=DataM0.X_BT.index)
            y_pred[y_name + "_BestAlgo" + cc].name = y_name + "_BestAlgo" + cc



    def Selection_Variable(self, load_mode=False):

        learn_sel = Learnings(learns=self)
        y_name = self.DataM.Y.name
        for algo_name, algo in self.algo_dict.items():

            # Selection de variable
            save_file = self.save_path + 'Selected_Variables_' + y_name + '_' + algo_name + '.pkl'
            load_file = self.load_path + 'Selected_Variables_' + y_name + '_' + algo_name + '.pkl'

            if load_mode and os.path.exists(self.load_path) and os.path.isfile(load_file):
                learn_sel.algo_dict[algo_name].Selected_Col = pickle.load(open(load_file, 'rb'))

            else:
                learn_sel.algo_dict[algo_name].Selected_Col = algo.Variable_S(self.DataM, self.mypath, algo_name)

                # Sauvgarde de l'object self avec pickle
                if not os.path.exists(self.save_path):
                    pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

                with open(save_file, 'wb') as output:
                    pickle.dump(learn_sel.algo_dict[algo_name].Selected_Col, output, pickle.HIGHEST_PROTOCOL)

            # if load_mode is False:
            #     learn_sel.algo_dict[algo_name].Selected_Col = algo.Variable_S(self.DataM, self.mypath, algo_name)
            #     # Sauvgarde de l'object self avec pickle
            #     if not os.path.exists(self.mypath0 + 'Pickle\\'):
            #         pathlib.Path(self.mypath0 + 'Pickle\\').mkdir(parents=True, exist_ok=True)
            #
            #     with open(self.mypath0 + "Pickle\\" + 'Selected_Variables_' + y_name + '_' + algo_name + '.pkl', 'wb') as output:
            #         pickle.dump(learn_sel.algo_dict[algo_name].Selected_Col, output, pickle.HIGHEST_PROTOCOL)
            # else:
            #     the_path = self.mypath0 + "Pickle\\" + 'Selected_Variables_' + y_name + '_' + algo_name + '.pkl'
            #     if os.path.isfile(the_path):
            #         learn_sel.algo_dict[algo_name].Selected_Col = pickle.load(open(the_path, 'rb'))
            #     else:
            #         learn_sel.algo_dict[algo_name].Selected_Col = algo.Variable_S(self.DataM, self.mypath, algo_name)
            #         # Sauvgarde de l'object self avec pickle
            #         if not os.path.exists(self.mypath0 + 'Pickle\\'):
            #             pathlib.Path(self.mypath0 + 'Pickle\\').mkdir(parents=True, exist_ok=True)
            #
            #         with open(self.mypath0 + "Pickle\\" + 'Selected_Variables_' + y_name + '_' + algo_name + '.pkl', 'wb') as output:
            #             pickle.dump(learn_sel.algo_dict[algo_name].Selected_Col, output, pickle.HIGHEST_PROTOCOL)


        return learn_sel

    def TrainTestGenerator(self):
        for train, test in self.kfold.split(self.DataM.X.values, self.strat):
            yield train, test
