from pandas import read_csv, to_datetime, DataFrame
import glob
import os
import re
import pandas as pd
from functools import reduce
import pathlib
import numpy as np
from EtudeBackTest import EtudeBackTest
from Analyse_BT import BackTest

class Analyse:

    def __init__(self, mypath=os.path.abspath("").replace("\\Code", "\\res\\")):
        self.mypath = mypath

        # self.strategies = read_csv(mypath + "Strategies.csv", sep=",")

        # self.strategies = glob.glob(mypath + '*.csv')
        # self.strategies = glob.glob(mypath + '*\\')
        # self.strategies = list(map(lambda x: x.replace(mypath, ''), self.strategies))

        self.strategies = os.listdir(mypath)
        self.backtest = list()

        #self.pred= dict()
        #self.weight = dict()
        #self.dict_1y = dict()
        #self.metric = list()
        self.Y_list = list()
        self.parametre_list = list()

    def Load(self, poidsmax=0.25):

        #path_hp = list(self.strategies["Nom Strategie"].loc[self.strategies["PoidsMax"] == poidsmax])

        # regex de date pour les repertoires
        reg = re.compile("^[0-9]{4}-(0?[1-9]|1[012])-(0?[1-9]|[12][0-9]|3[01])$")

        for p_hp0 in self.strategies:
            p_hp = self.mypath + p_hp0
            path_model = [name for name in os.listdir(p_hp)]

            for p_m0 in path_model:
                bt = BackTest()

                bt.Y_name = p_m0
                bt.Parametre_name = p_hp0

                if p_m0 not in self.Y_list:
                    self.Y_list.append(p_m0)

                if p_hp0 not in self.parametre_list:
                    self.parametre_list.append(p_hp0)

                p_m = p_hp + '\\' + p_m0

                for files_name in glob.glob(p_m + '\\' + '*.csv'):
                    ng = files_name.replace(p_m + "\\", "").replace(".csv", "")

                    if ng.replace(p_m0 + '--', '') in ['BackTest', 'Predictions', 'Poids']:
                        A = read_csv(files_name, sep=",")
                        if reg.match(A.iloc[0, 0]) is not None:
                            A.iloc[:, 0] = to_datetime(A.iloc[:, 0])
                        A.index = A.iloc[:, 0]

                        if ng.replace(p_m0 + '--', '') == 'BackTest':
                            bt.Valo = A.drop(A.columns[0], axis=1)

                        elif ng.replace(p_m0 + '--', '') == 'Predictions':
                            bt.Prediction = A.drop(A.columns[0], axis=1)

                        elif ng.replace(p_m0 + '--', '') == 'Poids':
                            bt.Weight = A.drop(A.columns[0], axis=1)

                self.backtest.append(bt)


                    #if ng.replace(p_m0 + '--', '') in ['BackTest', 'Predictions', 'Poids']:
                    #
                    #   A = read_csv(files_name, sep=",")
                    #   if reg.match(A.iloc[0, 0]) is not None:
                    #       A.iloc[:, 0] = to_datetime(A.iloc[:, 0])
                    #   A.index = A.iloc[:, 0]
                    #
                    #   if ng == p_m0 + '--BackTest':
                    #       self.backtest[p_hp0 + '--' + p_m0] = A.drop(A.columns[0], axis=1)
                    #
                    #   elif ng == p_m0 + '--Predictions':
                    #       self.pred[p_hp0 + '--' + p_m0] = A.drop(A.columns[0], axis=1)
                    #
                    #   elif ng == p_m0 + '--Poids':
                    #       self.weight[p_hp0 + '--' + p_m0] = A.drop(A.columns[0], axis=1)
                    #
                    #   else:
                    #       self.dict_1y[p_hp0 + '--' + p_m0 + '--' + ng] = A.drop(A.columns[0], axis=1)
                    #
                    #   if ng not in self.metric:
                    #       self.metric.append(ng)
                    #
                    #   if p_m0 not in self.contract_list:
                    #       self.contract_list.append(p_m0)
                    #
                    #   if p_hp0 not in self.parametre_list:
                    #       self.parametre_list.append(p_hp0)


    def agregation(self):

        for ct in self.Y_list:

            reg0 = re.compile("^[A-Za-z]*_[0-9]*d_")
            ct0 = reg0.sub('', ct)

            #bt_list = list(self.backtest.keys())
            reg = re.compile("^([a-z]|_|-|[0-9])*" + ct)

            y_bt = list(filter(lambda k: reg.match(k.Y_name) is not None, self.backtest))

            #y_bt = [self.backtest.get(k) for k in bt_list_filtered]

            y = y_bt[0].Valo
            y.columns = y_bt[0].Parametre_name + y.columns
            for x in y_bt[1:]:
                y = pd.merge(x.Valo, y, left_index=True, right_index=True, how='outer')

            #aa = reduce(lambda x, y: pd.merge(x.Valo, y.Valo, left_index=True, right_index=True, how='outer'),y_bt)

            y_bt = dict(zip(bt_list_filtered, y_bt))

            y_weight = [self.weight.get(k) for k in bt_list_filtered]
            y_weight = dict(zip(bt_list_filtered, y_weight))

            y_pred = [self.pred.get(k) for k in bt_list_filtered]
            y_pred = dict(zip(bt_list_filtered, y_pred))



            etude = EtudeBackTest(y_bt, y_weight, y_pred, ct0, self.mypath)
            etude.etudeBackTest()

        for ct in self.Y_list:

            reg0 = re.compile("^[A-Za-z]*_[0-9]*d_")
            ct0 = reg0.sub('', ct)

            for m in self.metric:

                reg = re.compile("^([a-z]|_|-|[0-9])*__" + ct + "__" + m)
                ll = list()
                for k, v in self.dict_1y.items():
                    if reg.match(k):

                        # Correction du nom des colonnes
                        p = k.replace(ct + "__" + m, "")
                        v.columns = list(map(lambda x: x.replace(ct, ct + "__" + p), v.columns))

                        if ll.__len__() > 0:
                            try:
                                v = v.drop(ct0, axis=1)
                            except Exception as Ex:
                                v = v.drop('0', axis=1)
                        ll.append(v)

                df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), ll)

                if m == 'BackTest_iuiu':
                    bt = df
                    p = df.loc[:, ct0]
                    r0 = p / p.shift(1) - 1

                if m == 'Poids_iuiu':
                    l0 = list()

                    for c in df.columns.values.tolist():
                        l0.append(bt.loc[:, c].to_frame())

                        for _ in range(10):

                            po = pd.DataFrame(data=np.random.choice(df.loc[:, c], df.shape[0]), index=df.index)

                            # Return + 1
                            r = r0*po + 1

                            # Ajout d'une ligne avec 100!
                            r.iloc[0] = 100

                            # Cummule des returns quotidiens pour calculer un prix fictif
                            r = r.cumprod(axis=0)

                            l0.append(r)

                        p0 = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), l0)
                        p0.to_csv(self.mypath + "\\Globaux\\" + ct + "\\" + c + "_random.csv")


                if m!= 'Stat_globales':
                    rank = df.rank(axis=1, method='first')

                    # Normalisation du rang entre 0 et 1 : 1=> Classement élevé (bonne perf!)
                    quantile = rank.quantile([0.25, 0.5, 0.75]).applymap(lambda x: x/df.shape[1])

                if not os.path.exists(self.mypath + "\\Globaux\\" + ct + "\\"):
                    pathlib.Path(self.mypath + "\\Globaux\\" + ct + "\\").mkdir(parents=True, exist_ok=True)

                df.to_csv(self.mypath + "\\Globaux\\"  + ct + "\\" + m + "_serie.csv")

                if m != 'Stat_globales':
                    rank.to_csv(self.mypath + "\\Globaux\\"  + ct + "\\" + m + "_rank.csv")
                    quantile.to_csv(self.mypath + "\\Globaux\\"  + ct + "\\" + m + "_quantile.csv")
