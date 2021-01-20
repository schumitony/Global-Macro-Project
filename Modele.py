from Learning import Learnings


from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from math import ceil
import os
from os import listdir
from os.path import join
import pickle
from functools import reduce
from LogFiles import log

import numpy as np
import pandas as pd


class Modele:

    def __init__(self, DataM, mypath, subpath, y, list_model, refit, weight_f, cv, save_path, load_path, centrage=None):

        self.list_model = list_model
        # self.list_refit = [refit for x in list_model]
        self.list_refit = refit

        self.kfold = []
        self.DataM = DataM.copy()
        self.algo = list()
        self.mypath0 = mypath
        self.mypath = mypath + subpath
        self.save_path = mypath + save_path
        self.load_path = mypath + load_path

        self.weight_f = weight_f
        self.centrage = centrage

        self.cv = cv

        # Centrage de la variable Y sur la moyenne passée
        if self.centrage == 'Global':
            self.DataM.Y = self.DataM.Y.apply(lambda x: x - x.mean())
        elif self.centrage == 'Glisse':
            self.DataM.Y = self.DataM.Y.apply(lambda x: x - x.rolling(window=self.DataM.Y.shape[0], min_periods=np.min([260, len(self.DataM.Y)])).mean())

        # Lag Prediction ( Correspond à l'horizon du return predit : il faut exclure la periode h de l'apprentissage)
        x = list(filter(lambda x: x.Nom == y, self.DataM.ListDataFrame0))[0]
        #self.Pred_lag = ceil(x.h / x.Freqence)
        self.Pred_lag = int(x.idx_h)


        # Selection de la variable Y
        self.DataM.Y = self.DataM.Y[y]
        self.DataM.Y_Out = self.DataM.Y_Out[y]
        self.DataM.Y_BT = self.DataM.Y_BT[y]

        # Réduction de l'historique de façon a optimiser la profondeur d'historique et le nombre de variable.
        nb_nan = self.DataM.X.notna().sum(axis=0).apply(lambda x: np.subtract(x, self.DataM.Y.notna().sum(axis=0)))

        # On calcul le nombre de série ayant un historique complet sur l'intervalle [Date de départ de Y + i jours; Fin de l'historique]
        aux = list()
        for i in range(20, 270*5, 20):
            aux.append((nb_nan > -i).sum() / nb_nan.shape[0])
        aux = pd.DataFrame(data=aux)

        iselc = (1 + aux[aux.iloc[:, 0] > 0.8].index[0]) * 20

        # Date de départ de Y
        StartDate = self.DataM.Y[self.DataM.Y.notna()].index[iselc]

        # Selection des dates en fonction de la date trouver précedement
        self.DataM.Y_Out = self.DataM.Y_Out.loc[self.DataM.Y_Out.index >= StartDate]
        self.DataM.Y = self.DataM.Y.loc[self.DataM.Y.index >= StartDate]

        self.DataM.X = self.DataM.X.loc[self.DataM.X.index >= StartDate, :]
        self.DataM.X_Out = self.DataM.X_Out.loc[self.DataM.X_Out.index >= StartDate, :]

        # Suppression de date de la fin d'historique correspondant a l'horizon du return (h)
        self.DataM.Y = self.DataM.Y.iloc[0:-self.Pred_lag]
        self.DataM.Y_Out = self.DataM.Y_Out.iloc[0:-self.Pred_lag]

        self.DataM.X = self.DataM.X.iloc[0:-self.Pred_lag, :]
        self.DataM.X_Out = self.DataM.X_Out.iloc[0:-self.Pred_lag, :]

        # Suppression des colonnes contenant encore des nans!
        nb_nan = self.DataM.X.notna().all(axis=0)

        # Suppression des series ayant une proportion de Nan trop elevee au debut
        Col = self.DataM.X.columns[nb_nan]
        self.DataM.X = self.DataM.X.loc[:, Col]
        self.DataM.X_Out = self.DataM.X_Out.loc[:, Col]
        self.DataM.X_BT = self.DataM.X_BT.loc[:, Col]

        self.DataM.ListDataFrame0 = [x for x in self.DataM.ListDataFrame0 if x.Nom in Col]

        # On selectionne i tel que le nombre de de série disponible soit au mon de 80%

        # Lag maximal
        self.H_idx_max = ceil(max([x.idx_h for x in self.DataM.ListDataFrame0]))

    def kfold_strategie(self, n_fold=4):

        nn = self.DataM.X.index.shape[0]
        k, m = divmod(nn, n_fold)

        if self.cv == 'blend_cv':
            for i in range(n_fold):
                testIndices = list(range(i * k + min(i, m), (i + 1) * k + min(i + 1, m)))

                x = list(range(0, max(0, min(nn, i * k + min(i, m) - (self.Pred_lag + 1)))))
                y = list(range(max(0, (i + 1) * k + min(i + 1, m) + self.H_idx_max + 1), self.DataM.X.index.shape[0]))
                trainIndices = x + y

                if testIndices.__len__() != 0 and trainIndices.__len__() != 0:
                    self.kfold.append((np.array(trainIndices), np.array(testIndices)))

        if self.cv == 'blend_test_cv':
            for i in range(n_fold-1):
                trainIndices = list(range(i * k + min(i, m), (i + 2) * k + min(i + 2, m)))

                x = list(range(0, max(0, min(nn, i * k + min(i, m) - (self.H_idx_max + 1)))))
                y = list(range(max(0, (i + 2) * k + min(i + 2, m) + self.Pred_lag + 1), self.DataM.X.index.shape[0]))
                testIndices = x + y

                if testIndices.__len__() != 0 and trainIndices.__len__() != 0:
                    self.kfold.append((np.array(trainIndices), np.array(testIndices)))

                    self.DataM.Y.iloc[testIndices].to_csv(os.path.abspath("").replace("\\Code", "\\TestY_" + str(i) + ".csv"))
                    self.DataM.Y.iloc[trainIndices].to_csv(os.path.abspath("").replace("\\Code", "\\TrainY_" + str(i) + ".csv"))


        elif self.cv == 'time_cv':
            for i in range(1, n_fold):

                trainIndices = list(range(0, max(0, min(nn, i * k - (self.Pred_lag + 1)))))
                testIndices = list(range(max(0, i * k), self.DataM.X.index.shape[0]))

                if testIndices.__len__() != 0 and trainIndices.__len__() != 0:
                    self.kfold.append((np.array(trainIndices), np.array(testIndices)))

    def Model_Reg(self, selection_variable=True, load_mode=False):
        """

        Modelisation retenue

        - Regression de Y sur les X

        Parameters
        ----------
        self : Instance de la classe
        list_model : Liste des modeles que l'on souhaite tester
        refit : Score d'evaluation des modèles
        selection_variable : execution routine de selection de variable
        load_mode : chargement des apprentissages existants

        Returns
        -------
        Return : X et Y

        """

        y_pred = dict()
        learn = dict()
        learn_sel = dict()

        res_file = log(path=self.mypath, nom=self.DataM.Y.name, create=True)

        y_name = self.DataM.Y.name
        learn.update({y_name: Learnings(DataM=self.DataM,
                                        list_model=self.list_model,
                                        list_refit=self.list_refit,
                                        kfold=self.kfold,
                                        weight_f=self.weight_f,
                                        mypath=self.mypath,
                                        mypath0=self.mypath0,
                                        save_path=self.save_path,
                                        load_path=self.load_path
                                        )})

        learn[y_name].load_model()
        learn[y_name].Learn_Algo(res_file, load_mode=load_mode)
        # learn[y_name].BlendModel()
        learn[y_name].Predict(y_pred)

        if selection_variable is True:
            learn_sel[y_name] = learn[y_name].Selection_Variable(load_mode=load_mode)
            learn_sel[y_name].Learn_Algo(res_file, TypeLearn=" selected variable", load_mode=load_mode)
            learn_sel[y_name].Predict(y_pred)

        return y_pred


   # Fonction de servant pas!!!!

    def Model_CoutMoy(self):
        """

        Modelisation retenue

        - On estime le nb de sinistre. Le modele est calé sur un jeu de train In.
        - Puis on fait les prévisions de nb de sinistre sur le jeu de train Out. On en déduit le cout moyen des sinistres
         (Cout / max(1,Prevision du nb de sinistre))
        - On estime le cout moyen des sinistres en calant un second modele sur le jeu de train Out
        - On peut alors calculer une prevision du cout des sinistre sur notre jeu de test en faisant le caculs suivant :
            prevision du cout des sinistre = prevision du nb de sinistre * prevision du cout moyen des sinistre

        Parameters
        ----------
        self : Instance de la classe

        Returns
        -------
        Return : X et Y

        """

        # Création des kfold
        self.kfold_strategie(n_fold=5)

        # n = 500
        # Xin, Xout, y_in, y_out = train_test_split(self.DataM.X.loc[0:n, :], self.DataM.Y.loc[0:n, :], test_size=0.5)
        Xin, Xout, y_in, y_out = train_test_split(self.DataM.X, self.DataM.Y, test_size=0.5)

        KmeanPred_in, _ = self.DataM.Kmean(Xin)
        KmeanPred_out, _ = self.DataM.Kmean(Xout)

        y_nb_sin_list = ['numtppd', 'numtpbi']
        y_cout_sin_list = ['inctppd', 'inctpbi']

        y = dict()
        y_test = dict()

        learn = dict()

        for nb_sin_name, cout_sin_name in zip(y_nb_sin_list, y_cout_sin_list):
            y.update({nb_sin_name: y_in.loc[:, nb_sin_name]})

            # Modelisation du nombre de sinistre de type 1
            list_model = ['RandForest', 'AdaB']
            learn.update({nb_sin_name: Learnings(X=Xin,
                                  Y=y[nb_sin_name],
                                  list_model=list_model,
                                  kfold=self.kfold,
                                  stratification=KmeanPred_in)})

            learn[nb_sin_name].load_model()
            learn[nb_sin_name].Learn_Algo()

            # fiting du meilleur modele
            learn[nb_sin_name].best_model_ever.fit(X=Xin, y=y[nb_sin_name])

            # Prediction de y out of sample
            y.update({nb_sin_name + '_out': y_out.loc[:, nb_sin_name]})
            y.update({nb_sin_name + '_out_predict': learn[nb_sin_name].best_model_ever.predict(Xout)})

            # Score de prediction
            # b = f1_score(y[nb_sin_name + '_out'], y[nb_sin_name + '_out_predict'], average='macro')

            # Cout Moyen des sinistre
            y.update({cout_sin_name + '_moy': pd.Series(np.divide(y_out.loc[:, cout_sin_name].values,
                                                                  np.array([max(x, 1) for x in y[nb_sin_name + "_out_predict"]])),
                                                        index=y[nb_sin_name + '_out'].index,
                                                        name=cout_sin_name + '_moy')})

            # Modelisation des couts
            self.DataM.Kmean(Xout)

            list_model = ['RandForestReg', 'AdaBReg']
            learn.update({cout_sin_name + '_moy': Learnings(X=Xout,
                                  Y=y[cout_sin_name + '_moy'],
                                  list_model=list_model,
                                  kfold=self.kfold,
                                  stratification=KmeanPred_out)})

            learn[cout_sin_name + '_moy'].load_model()
            learn[cout_sin_name + '_moy'].Learn_Algo()

            # Prevision sur le sample de test du nb de sinistre
            y_test.update({nb_sin_name: learn[nb_sin_name].best_model_ever.predict(self.DataM.X_test)})
            y_test.update({cout_sin_name + '_moy': learn[cout_sin_name + '_moy'].best_model_ever.predict(self.DataM.X_test)})
            y_test.update({cout_sin_name:  np.multiply(np.array([max(x, 1) for x in y_test[nb_sin_name]]), y_test[cout_sin_name + '_moy'])})

        Y_predic = pd.merge(pd.DataFrame(self.DataM.Y_PolNum), pd.DataFrame(np.add(y_test['inctppd'], y_test['inctpbi']), index=self.DataM.Y_PolNum.index), left_index=True, right_index=True)

        Y_predic.to_csv(self.DataM.path + '\\Model1.csv', encoding='utf-8', index=False)

    def Model_ZeroOne(self):
        """

        Modelisation retenue

        - On estime la presence de sinistre (0 ou 1 si nb sinistre > 0)!. Le modele est calé sur un jeu de train In.
        - Puis on fait les prévisions de la présence de sinistres sur le jeu de train Out. que l'on decompose en deux selon
          que l'on ait prévu ou non un sinistre
        - On modélise le cout des sinistres en calant un second modele sur les jeux de train Out "Zero" sinistre et "One" sinistre
        - On peut alors calculer une prevision du cout des sinistre sur notre jeu de test :
            - Previsiosn du nombre de sinostre
            - Selon le cas 0 ou 1 : prévision du cout des sinistres

        Parameters
        ----------
        self : Instance de la classe

        Returns
        -------
        Return : X et Y

        """

        # Création des kfold
        self.kfold_strategie()

        # n = 1000
        # Xin, X_O, y_in, y_out = train_test_split(self.DataM.X.loc[0:n, :], self.DataM.Y.loc[0:n, :], test_size=0.5)
        Xin, X_O, y_in, y_out = train_test_split(self.DataM.X, self.DataM.Y, test_size=0.5)

        KmeanPred_in, _ = self.DataM.Kmean(Xin)
        KmeanPred_out, _ = self.DataM.Kmean(X_O)

        y_nb_sin_list = ['numtppd', 'numtpbi']
        y_cout_sin_list = ['inctppd', 'inctpbi']

        y_PolNum = dict()
        y_test = dict()
        X_test = dict()

        y = dict()
        Xout = dict()

        learn = dict()

        for nb_sin_name, cout_sin_name in zip(y_nb_sin_list, y_cout_sin_list):
            # Transformation de notre varaible nb de sinistre en presence de sinistre (0 ou 1)!
            y.update({nb_sin_name: y_in.loc[:, nb_sin_name].apply(lambda x: min(x, 1))})

            # Modelisation de la présence de sinistre de type 1
            list_model = ['RandForest', 'AdaB']
            learn.update({nb_sin_name: Learnings(X=Xin,
                                  Y=y[nb_sin_name],
                                  list_model=list_model,
                                  kfold=self.kfold,
                                  stratification=KmeanPred_in)})

            learn[nb_sin_name].load_model()
            learn[nb_sin_name].Learn_Algo()

            # fiting du meilleur modele
            learn[nb_sin_name].best_model_ever.fit(X=Xin, y=y[nb_sin_name])

            # Prediction de y out of sample
            y.update({nb_sin_name + '_out': y_out.loc[:, nb_sin_name]})
            y.update({nb_sin_name + '_out_predict': pd.Series(learn[nb_sin_name].best_model_ever.predict(X_O),
                                                              index=X_O.index)})

            # Score de prediction
            # b = f1_score(y[nb_sin_name + '_out'], y[nb_sin_name + '_out_predict'], average='macro')

            # On test la taille des groupes. Si l'une d'elle est trop faible on ne fait pas l'analyse par groupe Zero et One
            if 10 < (y[nb_sin_name + '_out_predict'] == 0).sum() < y[nb_sin_name + '_out_predict'].shape[0] - 10:

                # Decoupage de notre train out entre ceux ayant une prévision avec des sinistres 1 et les sans sinistres 0
                Xout.update({'_Zero': X_O.loc[y[nb_sin_name + '_out_predict'].loc[:] == 0, :]})
                Xout.update({'_One': X_O.loc[y[nb_sin_name + '_out_predict'].loc[:] == 1, :]})

                y.update({cout_sin_name + '_Zero': y_out.loc[y[nb_sin_name + '_out_predict'].loc[:] == 0, cout_sin_name]})
                y.update({cout_sin_name + '_One':  y_out.loc[y[nb_sin_name + '_out_predict'].loc[:] == 1, cout_sin_name]})

                # Modelisation des couts
                y_o_list = [y[cout_sin_name + '_Zero'], y[cout_sin_name + '_One']]
                for Xi, Yi, name in zip(list(Xout.values()), y_o_list, Xout.keys()):

                    list_model = ['RandForestReg', 'AdaBReg']
                    learn.update({cout_sin_name + name: Learnings(X=Xi,
                                          Y=Yi,
                                          list_model=list_model,
                                          kfold=self.kfold)})

                    learn[cout_sin_name + name].load_model()
                    learn[cout_sin_name + name].Learn_Algo()

                # Prevision sur le sample de test du nb de sinistre
                y_test.update({nb_sin_name: learn[nb_sin_name].best_model_ever.predict(self.DataM.X_test)})

                # Decoupage de notre test out entre ceux ayant une prévision avec des sinistres 1 et les sans sinistres 0
                X_test.update({cout_sin_name + '_Zero': self.DataM.X_test.loc[y_test[nb_sin_name] == 0, :]})
                X_test.update({cout_sin_name + '_One':  self.DataM.X_test.loc[y_test[nb_sin_name] == 1, :]})

                y_PolNum.update({cout_sin_name + '_Zero': self.DataM.y_PolNum[y_test[nb_sin_name] == 0]})
                y_PolNum.update({cout_sin_name + '_One': self.DataM.y_PolNum[y_test[nb_sin_name] == 1]})

                # Prediction du cout des sinistres sur le jeu test selon le modèle des nb de sinistre à 0 et 1
                y_test.update({cout_sin_name + '_Zero': learn[cout_sin_name + '_Zero'].best_model_ever.predict(X_test[cout_sin_name + '_Zero'])})
                y_test.update({cout_sin_name + '_One':  learn[cout_sin_name + '_One'].best_model_ever.predict(X_test[cout_sin_name + '_One'])})

                # Concatenation des prévisions Zero et One
                y_test.update({cout_sin_name: np.concatenate((y_test[cout_sin_name + '_Zero'], y_test[cout_sin_name + '_One']), axis=0)})

                # Concatenation des numeros de police Zero et One
                y_aux = np.concatenate((y_PolNum[cout_sin_name + '_Zero'], y_PolNum[cout_sin_name + '_One']), axis=0)

                # Creation de la série des predictions indexé par le num de police
                y_test.update({cout_sin_name: pd.Series(y_test[cout_sin_name], index=y_aux)})

            else:

                KmeanPred, _ = self.DataM.Kmean(self.DataM.X)

                list_model = ['RandForestReg', 'AdaBReg']
                learn.update({cout_sin_name: Learnings(X=self.DataM.X,
                                                       Y=self.DataM.Y[cout_sin_name],
                                                       list_model=list_model,
                                                       kfold=self.kfold,
                                                       stratification=KmeanPred)})

                learn[cout_sin_name].load_model()
                learn[cout_sin_name].Learn_Algo()

                # Prediction de notre jeu de Test
                y_test.update({cout_sin_name: learn[cout_sin_name].best_model_ever.predict(self.DataM.X_test)})

                # Creation de la série des predictions indexé par le num de police
                y_test.update({cout_sin_name: pd.Series(y_test[cout_sin_name], index=self.DataM.y_PolNum)})



        # On additionne les deux couts de sinistre par numero de police
        # (d'ou la creation au prealable de series indéxées sur le numero de police
        Y_predic = y_test['inctppd'].add(y_test['inctpbi'])
        Y_predic.name = 'Predictions'
        Y_predic.index.name = 'Id'

        Y_predic.to_csv(self.DataM.path + '\\Model2.csv', encoding='utf-8', index=True, header=True)

    def Model_CoutMoy2(self, Cut_Xvalue=False, Stratification=False, n_fold=4, selection_variable=True):
        """

        Modelisation retenue

        - On estime le cout des deux type de sinistre independament puis on somme les prévisions!

        Parameters
        ----------
        self : Instance de la classe

        Returns
        -------
        Return : X et Y

        """

        # Création des kfold
        self.kfold_strategie(n_fold)

        y_nb_sin_list = ['numtppd', 'numtpbi']
        y_cout_sin_list = ['inctppd', 'inctpbi']

        y = dict()
        Mu = dict()
        y_test = dict()
        learn = dict()
        learn_sel = dict()

        if Stratification is True:
            KmeanPred, _ = self.DataM.Kmean(self.DataM.X_OHE)
            Strat = '_Strat'
        else:
            KmeanPred = None
            Strat = ''

        for nb_sin_name, cout_sin_name in zip(y_nb_sin_list, y_cout_sin_list):

            # Cout Moyen des sinistre
            y[cout_sin_name + '_moy'] = pd.Series(data=np.divide(self.DataM.Y_All.loc[:, cout_sin_name].values,
                                                                 np.array([max(x, 1) for x in self.DataM.Y_All.loc[:, nb_sin_name].values])),
                                                  index=self.DataM.Y_All.index,
                                                  name=cout_sin_name + '_moy')

            Mu[cout_sin_name] = y[cout_sin_name + '_moy'][y[cout_sin_name + '_moy'] > 0].mean()

            self.DataM.Y_All[nb_sin_name] = self.DataM.Y_All[nb_sin_name].apply(lambda x: min(x, 3))
            self.DataM.Y_Out_All[nb_sin_name] = self.DataM.Y_Out_All[nb_sin_name].apply(lambda x: min(x, 3))

            self.DataM.Y = self.DataM.Y_All[nb_sin_name]
            self.DataM.Y_Out = self.DataM.Y_Out_All[nb_sin_name]

            text_file = open(self.mypath + '\\Results' + '_' + nb_sin_name + Strat + '.txt', "w")

            # list_model = ['RandForestReg', 'AdaBReg', 'GBoostMReg', 'XGBRegressor']
            # list_refit = ['f1_macro', 'f1_macro']
            # list_refit = ['neg_mean_absolute_error', 'neg_mean_absolute_error']
            list_model = ['XGBClassifier', 'RandForest']
            list_refit = ['f1_macro', 'f1_macro']

            learn.update({nb_sin_name: Learnings(DataM=self.DataM,
                                                 list_model=list_model,
                                                 list_refit=list_refit,
                                                 kfold=self.kfold,
                                                 stratification=KmeanPred)})

            learn[nb_sin_name].load_model()
            learn[nb_sin_name].Learn_Algo(text_file)
            learn[nb_sin_name].Predict_CoutMoy2(y_test, Mu[cout_sin_name])

            if selection_variable is True:
                learn_sel[nb_sin_name] = learn[nb_sin_name].Selection_Variable(text_file)
                learn_sel[nb_sin_name].Learn_Algo(text_file, " selected variable")
                learn_sel[nb_sin_name].Predict_CoutMoy2(y_test, Mu[cout_sin_name])

            text_file.close()



        Modele.Predit_to_csv(y_test, list_model, "", Strat, self.DataM.y_PolNum, self.mypath, v1='numtppd_', v2='numtpbi_')
        Modele.Predit_to_csv(y_test, list_model, "", Strat, self.DataM.y_PolNum, self.mypath, v1='numtppd_', v2='numtpbi_', cc="_Selected")

    def Model_Direct(self, Cut_Xvalue=False, Stratification=False, n_fold=4, selection_variable=True):
        """

        Modelisation retenue

        - On estime le cout des deux type de sinistre independament puis on somme les prévisions!

        Parameters
        ----------
        self : Instance de la classe

        Returns
        -------
        Return : X et Y

        """

        # Création des kfold
        self.kfold_strategie(n_fold)

        y_nb_sin_list = ['numtppd', 'numtpbi']
        y_cout_sin_list = ['inctppd', 'inctpbi']

        y_test = dict()
        learn = dict()
        learn_sel = dict()

        if Stratification is True:
            KmeanPred, _ = self.DataM.Kmean(self.DataM.X_OHE)
            Strat = '_Strat'
        else:
            KmeanPred = None
            Strat = ''

        for nb_sin_name, cout_sin_name in zip(y_nb_sin_list, y_cout_sin_list):
            if Cut_Xvalue is False:
                self.DataM.Y = self.DataM.Y_All[cout_sin_name]
                self.DataM.Y_Out = self.DataM.Y_Out_All[cout_sin_name]
                Xv = ''
            else:
                c = self.DataM.Y_All[cout_sin_name].mean() + 2 * self.DataM.Y_All[cout_sin_name].std()
                self.DataM.Y = self.DataM.Y_All[cout_sin_name].apply(lambda x: min(x, c))
                self.DataM.Y_Out = self.DataM.Y_Out_All[cout_sin_name].apply(lambda x: min(x, c))
                Xv = '_Cut_Xvalue'

            text_file = open(self.mypath + '\\Results' + '_' + cout_sin_name + Xv + Strat + '.txt', "w")

            # list_model = ['RandForestReg', 'AdaBReg', 'GBoostMReg', 'XGBRegressor']
            # list_refit = ['f1_macro', 'f1_macro']
            # list_refit = ['neg_mean_absolute_error', 'neg_mean_absolute_error']
            # list_model = ['XGBRegressor', 'RandForestReg']
            # list_refit = ['neg_mean_squared_error', 'neg_mean_squared_error']
            list_model = ['SVR']
            list_refit = ['neg_mean_squared_error']

            learn.update({cout_sin_name: Learnings(DataM=self.DataM,
                                                   list_model=list_model,
                                                   list_refit=list_refit,
                                                   kfold=self.kfold,
                                                   stratification=KmeanPred)})

            learn[cout_sin_name].load_model()
            learn[cout_sin_name].Learn_Algo(text_file)
            learn[cout_sin_name].Predict(y_test)

            if selection_variable is True:
                learn_sel[cout_sin_name] = learn[cout_sin_name].Selection_Variable(text_file)
                learn_sel[cout_sin_name].Learn_Algo(text_file, " selected variable")
                learn_sel[cout_sin_name].Predict(y_test)

            text_file.close()

        Modele.Predit_to_csv(y_test, list_model, Xv, Strat, self.DataM.y_PolNum, self.mypath)
        Modele.Predit_to_csv(y_test, list_model, Xv, Strat, self.DataM.y_PolNum, self.mypath, cc="_Selected")


    @staticmethod
    def Predit_to_csv(y_test, list_model, Xv, Strat, y_PolNum, mypath, v1='inctppd_', v2='inctpbi_', cc=""):

        # On additionne les deux couts de sinistre par numero de police
        # (d'ou la creation au prealable de series indéxées sur le numero de police
        y0 = list()
        for m1 in list_model:
            for m2 in list_model:
                Y_predic = y_test[v1 + m1 + cc].add(y_test[v2 + m2 + cc])
                y0.append(Y_predic)

                Y_predic.name = 'Predictions'
                Y_predic.index.name = 'Id'

                Y_predic.to_csv(mypath + '\\Model_Direct_' + m1 + '_' + m2 + Xv + Strat + cc + '.csv', encoding='utf-8',
                                index=True, header=True)

        Y_predic_All_modele = np.stack(y0)

        # Prediction des meilleurs modele
        Y_predic = y_test[v1 + cc].add(y_test[v2 + cc])
        Y_predic.name = 'Predictions'
        Y_predic.index.name = 'Id'

        Y_predic.to_csv(mypath + '\\Model_Direct' + Xv + Strat + cc + '.csv', encoding='utf-8', index=True, header=True)

        # Predicteur melangé par Regression lineaire
        # Y_predic_All_modele = None
        # y0 = list()
        # for m1 in list_model:
        #     for m2 in list_model:
        #         # Prediction sur la base de train pour chaque modele
        #         y0.append(learn['inctppd'].best_model[m1].predict(self.DataM.X) + learn['inctpbi'].best_model[m2].predict(self.DataM.X))

        # Y_predic_All_modele = pd.DataFrame(np.stack(y0, axis=-1))

        # lr = LinearRegression().fit(Y_predic_All_modele, self.DataM.Y['inctpbi'] + self.DataM.Y['inctppd'])

        # Y_predic = np.matmul(np.matrix(lr.coef_), np.transpose(Y_predic_All_modele))
        Y_predic_All_modele = Y_predic_All_modele.mean(axis=0)
        Y_predic_All_modele = pd.Series(Y_predic_All_modele, index=y_PolNum)
        Y_predic_All_modele.name = 'Predictions'
        Y_predic_All_modele.index.name = 'Id'

        Y_predic_All_modele.to_csv(mypath + '\\Model_Direct_Blend' + Xv + Strat + '.csv', encoding='utf-8',
                                   index=True,
                                   header=True)

    def LoadModel(self):

        pkl_files = [join(self.mypath, f) for f in listdir(self.mypath) if '.pkl' in f]
        modele = [pickle.load(open(o, 'rb')) for o in pkl_files]


        y1 = modele[0].predict(self.DataM.X) + modele[2].predict(self.DataM.X)
        y2 = modele[1].predict(self.DataM.X) + modele[3].predict(self.DataM.X)
        Xy = pd.DataFrame(np.stack((y1, y2), axis=-1))

        lr = LinearRegression().fit(Xy, self.DataM.Y['inctpbi'] + self.DataM.Y['inctppd'])

    def Model_Zero(self, Cut_Xvalue=False, Stratification=False):
        """

        Modelisation retenue

        - On estime la presence de sinistre (0 ou 1 si nb sinistre > 0)!. Le modele est calé sur un jeu de train In.
        - Puis on fait les prévisions de nb de sinistre sur le jeu de train Out. On en déduit le cout moyen des sinistres
         (Cout / max(1,Prevision du nb de sinistre))
        - On estime le cout moyen des sinistres en calant un second modele sur le jeu de train Out
        - On peut alors calculer une prevision du cout des sinistre sur notre jeu de test en faisant le caculs suivant :
            prevision du cout des sinistre = prevision du nb de sinistre * prevision du cout moyen des sinistre

        Parameters
        ----------
        self : Instance de la classe

        Returns
        -------
        Return : X et Y

        """

        # Création des kfold
        self.kfold_strategie()

        y_nb_sin_list = ['numtppd', 'numtpbi']
        y_cout_sin_list = ['inctppd', 'inctpbi']

        y_PolNum = dict()
        X_Classe = dict()
        Y_Classe = dict()
        y_test = dict()
        learnClasse = dict()
        learn = dict()

        if Stratification is True:
            KmeanPred, _ = self.DataM.Kmean(self.DataM.X)
            Strat = '_Strat'
        else:
            KmeanPred = None
            Strat = ''

        for nb_sin_name, cout_sin_name in zip(y_nb_sin_list, y_cout_sin_list):

            # Modelisation de la presence de sinistre ou non!
            # Transformation de notre varaible nb de sinistre en presence de sinistre (0 ou 1)!
            Y_train0 = self.DataM.Y[cout_sin_name].apply(lambda x: 0 if x == 0 else 1)

            # Modelisation de la présence de sinistre de type 1
            list_model = ['XGBClassifier', 'RandForest']
            learnClasse.update({cout_sin_name: Learnings(X=self.DataM.X,
                                Y=Y_train0,
                                list_model=list_model,
                                kfold=self.kfold,
                                stratification=KmeanPred)})

            learnClasse[cout_sin_name].load_model()
            learnClasse[cout_sin_name].Learn_Algo(path=self.mypath)


            # Prevision du meilleur algo pour chaque sinistre
            y_test.update({cout_sin_name + '_Classe': learnClasse[cout_sin_name].best_model_ever.predict(self.DataM.X)})

            # Decoupage de notre test out entre ceux ayant une prévision avec des sinistres 1 et les sans sinistres 0
            X_Classe.update({cout_sin_name + '_Classe_Zero': self.DataM.X.loc[y_test[cout_sin_name + '_Classe'] == 0, :]})
            X_Classe.update({cout_sin_name + '_Classe_One': self.DataM.X.loc[y_test[cout_sin_name + '_Classe'] == 1, :]})

            Y_Classe.update({cout_sin_name + '_Classe_Zero': self.DataM.Y[y_test[cout_sin_name + '_Classe'] == 0].loc[:,cout_sin_name]})
            Y_Classe.update({cout_sin_name + '_Classe_One': self.DataM.Y[y_test[cout_sin_name + '_Classe'] == 1].loc[:,cout_sin_name]})


            #Apprentissage du cout des sinistres sur les individus dont la prevision annonce des sinistres
            if Cut_Xvalue is False:
                Y_train = Y_Classe[cout_sin_name + '_Classe_One']
                Xv = ''
            else:
                c = self.DataM.Y[cout_sin_name].mean() + 2 * self.DataM.Y[cout_sin_name].std()
                Y_train = Y_Classe[cout_sin_name + '_Classe_One'].apply(lambda x: min(x, c))
                Xv = '_Cut_Xvalue'


            list_model = ['XGBRegressor', 'RandForestReg']
            learn.update({cout_sin_name: Learnings(X=X_Classe[cout_sin_name + '_Classe_One'],
                                                   Y=Y_train,
                                                   list_model=list_model,
                                                   kfold=self.kfold,
                                                   stratification=KmeanPred)})

            learn[cout_sin_name].load_model()
            learn[cout_sin_name].Learn_Algo(Xv, Strat, self.mypath)

            # Prediction de notre jeu de Test
            #Prediction de la classe 0 ou  1 sinistre et plus
            y_test.update({cout_sin_name + '_test_Classe': learnClasse[cout_sin_name].best_model_ever.predict(self.DataM.X_test)})

            # Decoupage de notre test out entre ceux ayant une prévision avec des sinistres 1 et les sans sinistres 0
            X_Classe.update({cout_sin_name + '_test_Classe_Zero': self.DataM.X_test.loc[y_test[cout_sin_name + '_test_Classe'] == 0, :]})
            X_Classe.update({cout_sin_name + '_test_Classe_One': self.DataM.X_test.loc[y_test[cout_sin_name + '_test_Classe'] == 1, :]})

            y_PolNum.update({cout_sin_name + '_Zero': self.DataM.y_PolNum[y_test[cout_sin_name + '_test_Classe'] == 0]})
            y_PolNum.update({cout_sin_name + '_One': self.DataM.y_PolNum[y_test[cout_sin_name + '_test_Classe'] == 1]})

            # for m, _ in learn[cout_sin_name].algo.items():
            #     # Prediction de notre jeu de Test
            #     y_test.update({cout_sin_name + '_' + m: learn[cout_sin_name].
            #                   best_model[m].predict(X_Classe[cout_sin_name + '_test_Classe_One'])})
            #
            #     # Creation de la série des predictions indexé par le num de police
            #     y_test.update(
            #         {cout_sin_name + '_' + m: pd.Series(y_test[cout_sin_name + '_' + m], index=y_PolNum[cout_sin_name + '_One'])})

            # Prediction de notre jeu de Test
            if X_Classe[cout_sin_name + '_test_Classe_One'].shape[0] > 5:
                y_test.update({cout_sin_name: learn[cout_sin_name].best_model_ever.predict(X_Classe[cout_sin_name + '_test_Classe_One'])})

                # Concatenation des prévisions Zero et One
                y_test.update({cout_sin_name: np.concatenate(
                    (np.zeros(y_PolNum[cout_sin_name + '_Zero'].shape), y_test[cout_sin_name]), axis=0)})

                # Concatenation des numeros de police Zero et One
                PolNum_aux = np.concatenate((y_PolNum[cout_sin_name + '_Zero'], y_PolNum[cout_sin_name + '_One']), axis=0)
            else:
                # Concatenation des prévisions Zero et One
                y_test.update({cout_sin_name: np.zeros(y_PolNum[cout_sin_name + '_Zero'].shape)})

                # Concatenation des numeros de police Zero et One
                PolNum_aux = y_PolNum[cout_sin_name + '_Zero']

            # Creation de la série des predictions indexé par le num de police
            y_test.update({cout_sin_name: pd.Series(y_test[cout_sin_name], index=PolNum_aux)})

        # Selection de variables

        # On additionne les deux couts de sinistre par numero de police
        # (d'ou la creation au prealable de series indéxées sur le numero de police
        # y0 = list()
        # for m1 in list_model:
        #     for m2 in list_model:
        #       Y_predic = y_test['inctppd_' + m1].add(y_test['inctpbi_' + m2])
        #       y0.append(Y_predic)

        #       Y_predic.name = 'Predictions'
        #       Y_predic.index.name = 'Id'

        #       Y_predic.to_csv(self.mypath + '\\Model_Direct_' + m1 + '_' + m2 + Xv + Strat + '.csv',
        #                            encoding='utf-8',
        #                           index=True, header=True)

        # Y_predic_All_modele = np.stack(y0)

        # Prediction des meilleurs modele
        Y_predic = y_test['inctppd'].add(y_test['inctpbi'])
        Y_predic.name = 'Predictions'
        Y_predic.index.name = 'Id'

        Y_predic.to_csv(self.mypath + '\\Model_Zero' + Xv + Strat + '.csv', encoding='utf-8', index=True,
                        header=True)

        # Predicteur melangé par Regression lineaire
        # Y_predic_All_modele = None
        # y0 = list()
        # for m1 in list_model:
        #     for m2 in list_model:
        #         # Prediction sur la base de train pour chaque modele
        #         y0.append(learn['inctppd'].best_model[m1].predict(self.DataM.X) + learn['inctpbi'].best_model[m2].predict(self.DataM.X))

        # Y_predic_All_modele = pd.DataFrame(np.stack(y0, axis=-1))

        # lr = LinearRegression().fit(Y_predic_All_modele, self.DataM.Y['inctpbi'] + self.DataM.Y['inctppd'])

        # Y_predic = np.matmul(np.matrix(lr.coef_), np.transpose(Y_predic_All_modele))
        # Y_predic_All_modele.mean(axis=0)
        # Y_predic_All_modele = pd.DataFrame(Y_predic_All_modele)
        # Y_predic_All_modele.name = 'Predictions'
        # Y_predic_All_modele.index.name = 'Id'

        # Y_predic_All_modele.to_csv(self.mypath + '\\Model_Direct_Blend' + Xv + Strat + '.csv', encoding='utf-8',
        #                            index=True,
        #                            header=True)

    def Predit_summed_to_csv(self, y_test, list_model, Strat, y_PolNum, mypath, v1='Sum', cc=""):

        # On additionne les deux couts de sinistre par numero de police
        # (d'ou la creation au prealable de series indéxées sur le numero de police
        y0 = list()
        y00 = list()
        for m1 in list_model:
            Y_predic = y_test[v1 + '_' + m1 + cc]
            Y_predic.name = 'Predictions'
            Y_predic.index.name = 'Id'

            y0.append(y_test[v1 + '_Out_' + m1 + cc])
            y00.append(Y_predic)
            Y_predic.to_csv(mypath + '\\Model_Direct_' + m1 + '_' + Strat + cc + '.csv', encoding='utf-8',
                            index=True, header=True)

        Y_predic_All_modele00 = pd.DataFrame(np.stack(y00, axis=-1))
        Y_predic_All_modele = pd.DataFrame(np.stack(y0, axis=-1))

        # Prediction des meilleurs modele
        Y_predic = y_test[v1 + cc]
        Y_predic.name = 'Predictions'
        Y_predic.index.name = 'Id'

        Y_predic.to_csv(mypath + '\\Model_Direct' + Strat + cc + '.csv', encoding='utf-8', index=True, header=True)

        # Predicteur melangé par Regression lineaire
        scoring = ["neg_mean_absolute_error", "neg_mean_squared_error"]
        p_grid = {"alpha": list(range(200, 20001, 200))}
        model = Ridge(fit_intercept=True)

        A = GridSearchCV(estimator=model, param_grid=p_grid, n_jobs=-1,
                         scoring=scoring, refit='neg_mean_squared_error', verbose=2)

        A.fit(Y_predic_All_modele, self.DataM.Y_Out)

        Yzz = A.predict(Y_predic_All_modele00)
        Yzz = pd.Series(Yzz, index=y_PolNum)
        Yzz.name = 'Predictions'
        Yzz.index.name = 'Id'

        Yzz.to_csv(mypath + '\\Model_Direct_Blend_Ridge' + Strat + '.csv', encoding='utf-8',
                   index=True,
                   header=True)

        # Y_predic = np.matmul(np.matrix(lr.coef_), np.transpose(Y_predic_All_modele))
        Y_predic_All_modele = Y_predic_All_modele.mean(axis=0)
        Y_predic_All_modele = pd.Series(Y_predic_All_modele, index=y_PolNum)
        Y_predic_All_modele.name = 'Predictions'
        Y_predic_All_modele.index.name = 'Id'

        Y_predic_All_modele.to_csv(mypath + '\\Model_Direct_Blend' + Strat + '.csv', encoding='utf-8',
                                   index=True,
                                   header=True)











