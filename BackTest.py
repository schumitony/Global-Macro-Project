from Modele import Modele
from EtudeBackTest import EtudeBackTest
from Utilitaire import Utilitaire as ut
import datetime as dt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import pickle
import copy
import pathlib
import shutil as sh
import glob
import re
import bisect

#import matplotlib.pyplot as plt

#Test idiot GitHub !!!!!!!

class BackTest:

    def __init__(self, DataM, mypath, y, price_name, strategie, list_model, refit, h, cv, max_weight,centrage,
                 pred_files="Predictions_Save"):

        self.DataM = DataM.copy()
        self.mypath = os.path.abspath("").replace("Program_ML\\Code", mypath + "\\" + strategie + "--maxp_"
                                                  + str(max_weight*100).replace('.0', '') + '--' + refit + '--' + cv + '\\')
        self.mypath = self.mypath + y + '\\'

        self.pred_files = pred_files
        # for e in list_model:
        #     self.pred_files = self.pred_files + '_' + e

        self.cv = cv

        self.y = y
        # self.weight_f = lambda x: getattr(BackTest, strategie)(x, max_weight)

        self.weight_f = getattr(self, strategie)
        self.max_weight = max_weight
        self.Price = pd.DataFrame
        self.Price_name = price_name
        self.h = h

        self.strategie = strategie
        self.list_model = list_model
        self.refit = refit
        self.centrage = centrage

        self.y_pred = list()

    def estime_bt(self):

        load_mode = True

        Bt_Duration = 3

        y_pred, dt0, dt00 = self.check_pred(Bt_Duration=Bt_Duration)

        while dt0 <= self.DataM.Data.index[-1]:

            # if load_date is not None:
            #     load_mode = (True if dt0 < dt.datetime.strptime(load_date, '%Y/%m/%d') else False)

            self.DataM.Decoupage(AbsoluteStart=dt00, EndTest=dt0, Bt_Duration=Bt_Duration, TypeDecoupage='Overlap')
            self.Price = self.DataM.Data_Price.loc[:, self.Price_name]

            # centrage = None if self.h == 0 else 'Glisse'

            M = Modele(DataM=self.DataM,
                       y=self.y,
                       list_model=self.list_model,
                       refit=self.refit,
                       weight_f=self.weight_f,
                       cv=self.cv,
                       mypath=self.mypath,
                       subpath=str(dt0.year) + "-" + str(dt0.month) + "\\",
                       save_path='Pickle\\LearnInProgress\\',
                       load_path='Pickle\\LearnCompleted\\',
                       centrage=self.centrage
                       )

            M.kfold_strategie()
            y_pred0 = M.Model_Reg(load_mode=load_mode)

            self.add_dict(y_pred, y_pred0)

            # Copie/Colle de WorkinProgress à WorkCompleted puis efface les apprentissages le dossier Work in Progress
            if not os.path.exists(M.load_path):
                pathlib.Path(M.load_path).mkdir(parents=True, exist_ok=True)

            for files_name in glob.glob(M.save_path + '*.pkl'):
                sh.copy2(files_name, M.load_path)
                os.remove(files_name)

            y_bt = self.estime_perf(y_pred)

            dt0 = dt0 + relativedelta(months=Bt_Duration)
            load_mode = False

        with open(self.mypath + 'Backtest.pkl', 'wb') as output:
            pickle.dump(y_bt, output, pickle.HIGHEST_PROTOCOL)

        return y_bt

    def weight_ls(self, return_pred):
        a = 80
        return self.max_weight*(np.exp(return_pred * a) - 1)/(np.exp(return_pred * a) + 1)

    def weight_ls_param_rescaling_Old(self, return_pred):
        a = min(300, 200*(0.01/np.std(return_pred)))

        return self.max_weight*(np.exp(return_pred * a) - 1)/(np.exp(return_pred * a) + 1)

    def weight_ls_param_rescaling(self, return_pred):
        # a = 200 corresponds à la transformation pour des previsions ayant
        #  un ecart type de 1% sur l'horizon de prevision

        # Ecart Type sur fenetre glissante de un an pour ne pas utiliser le future!
        std = pd.DataFrame(return_pred).rolling(window=260, min_periods=5).std()
        std[std.isna().values] = 0.01
        a = 200 * (0.01 / std)
        a[a > 300] = 300

        a = a.values.reshape(return_pred.shape)

        x = np.multiply(a, return_pred)
        return self.max_weight * (np.exp(x) - 1) / (np.exp(x) + 1)

    def check_pred(self, Bt_Duration):
        # Cette fonction renvoie l'historique des prevision si cette derniere on deja été produite, ainsi que la
        # dernière date de mise à jour du modèle

        Y = list(filter(lambda x: x.Nom == self.y, self.DataM.ListDataFrame0))[0].S
        i = np.where(np.isnan(Y) == False)

        dt00 = Y.index[i[0][0]] + relativedelta(years=2)
        qbegins = [dt.datetime(dt00.year, month, 1) + relativedelta(days=-1) for month in (1, 4, 7, 10)]\
                  + [dt.datetime(dt00.year + 1, 1, 1) + relativedelta(days=-1)]
        idx = bisect.bisect_left(qbegins, dt00)

        dt00 = qbegins[idx]

        # dt00 = dt.datetime.strptime('2005/12/31', '%Y/%m/%d')

        y_pred = dict()

        if os.path.isfile(self.mypath + "\\Predictions_Save\\" + self.pred_files + ".csv"):

            y_pred0 = self.ReadCsv()
            if y_pred0.__len__() == 0:
                return y_pred, dt00

            # Determination de la date correspondant au debut de la période d'utilisation du dernier calibrage
            dt_end = y_pred0.index[-1] + relativedelta(months=-Bt_Duration)
            while dt00 < dt_end:
                dt00 = dt00 + relativedelta(months=Bt_Duration)

            # Suppression des données du modèle actuel. On les génère à nouveau à partir de la sauvgarde du modèle
            y_pred0 = y_pred0.loc[y_pred0.index <= dt00, :]

            if y_pred0.__len__() > 0:
                for col in y_pred0.columns:
                    y_pred[col] = y_pred0[col].to_frame()

                dt0 = y_pred0.index[-1]
            else:
                dt0 = dt00
        else:
            dt0 = dt00

        return y_pred, dt0, dt00

    def ReadCsv(self, InModel=True):

        y_pred0 = pd.read_csv(self.mypath + "\\Predictions_Save\\" + self.pred_files + ".csv", sep=",")

        y_pred0.index = pd.to_datetime(y_pred0.iloc[:, 0], format='%Y-%m-%d')
        y_pred0.index.name = 'Date'
        y_pred0 = y_pred0.drop([y_pred0.columns[0]], axis=1)

        # Selection des colonnes des models que l'on cherche a estimer pour ne conservé que l'historique des
        # previsions de ces modèles
        if InModel:
            y_col = []
        else:
            y_col = list(y_pred0.columns)

        for m in self.list_model:
            if InModel:
                r = re.compile('.*' + m + '.*')
                y_col = y_col + list(filter(r.match, y_pred0.columns))
            else:
                r = re.compile('^(?!.*' + m + ').*$')
                y_col = list(filter(r.match, y_col))

        y_pred0 = y_pred0[y_col]

        # On ne retient que les lignes n'ayant pas de Nan
        NotnanIndex = y_pred0.isna().sum(axis=1) == 0
        y_pred0 = y_pred0.loc[NotnanIndex, :]

        if y_pred0.empty:
            y_pred0 = pd.DataFrame()

        return y_pred0

    def add_dict(self, y_pred, y_pred0):
        for k, v in y_pred0.items():

            if isinstance(v, pd.Series):
                v = v.to_frame(name=v.name)

            if k not in y_pred:
                y_pred[k] = v
            else:
                y_pred[k] = pd.concat((y_pred[k], v), axis=0)

        df = ut.dict_list_to_df(y_pred)

        if not os.path.exists(self.mypath + "//Predictions_Save"):
            pathlib.Path(self.mypath + "//Predictions_Save").mkdir(parents=True, exist_ok=True)
        else:
            y_pred = self.ReadCsv(False)
            df = pd.concat([df, y_pred], axis=1, join='outer')

        df.to_csv(self.mypath + "\\Predictions_Save\\" + self.pred_files + ".csv")

    def estime_perf(self, y_pred0):
        y_bt = dict()
        y_weight = dict()

        y_pred = copy.copy(y_pred0)

        y_pred[self.Price.name] = (self.Price.shift(-self.h) / self.Price - 1).to_frame()

        # Valorisation de la stratégie
        for k, pred in y_pred.items():

            actif = list()
            actif.append(100)
            q = list()

            if self.Price.name != k:
                x = pd.merge(pred, self.Price.to_frame(), left_index=True, right_index=True, how='inner')
                x = x.iloc[0:pred.shape[0]+1, :]

                self.Price = x.loc[:, self.Price.name]
                pred = x.loc[:, pred.columns[0]]

                # Backtesting
                w0_m = self.weight_f(pred)
                w_m = w0_m.rolling(window=self.h, min_periods=1).sum()
                #w_m = pred.apply(self.weight_f).rolling(window=self.h, min_periods=1).sum()
                w_m = w_m.map(lambda x: min(max(x, -self.max_weight), self.max_weight))
            else:
                # Pour l'investissement support, on simule une exposition fixe maximum
                w_m = pd.Series(data=np.ones(self.Price.shape) * self.max_weight, index=self.Price.index)

            for i in range(0, w_m.shape[0]-1):
                q.append(actif[i] * w_m[i] / self.Price[i])
                actif.append(actif[i] + q[i] * (self.Price[i+1] - self.Price[i]))

            # weight = np.cumsum(np.multiply(y_pred[k].apply(self.weight_f), self.DataM.Y_BT))
            y_weight[k] = w_m.to_frame(name=k)
            y_bt[k] = pd.DataFrame(data=actif, index=self.Price.index, columns=[k])

        etude = EtudeBackTest(y_bt, y_pred, y_weight, self.Price.name, self.y, self.mypath)
        etude.etudeBackTest()
        etude.all_to_csv()

        # plt.figure()
        # plt.plot(df, label=df.columns())
        # plt.legend(loc='best')

        # plt.savefig(self.mypath + 'BackTest.pdf')

        return y_bt