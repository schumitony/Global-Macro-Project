from Utilitaire import Utilitaire
import datetime as dt
import numpy as np
import pandas as pd
import re
import os

class EtudeBackTest:

    def __init__(self, y_bt, y_pred, y_weight, underlying, ul_h, mypath):

        self.underlying = underlying
        self.mypath = mypath
        self.y_bt = Utilitaire.to_frame(y_bt)
        self.y_pred = Utilitaire.to_frame(y_pred)
        self.y_weight = Utilitaire.to_frame(y_weight)
        self.ul_h = ul_h

        self.stat_global = pd.DataFrame()
        self.perf = dict()
        self.correl = dict()
        self.ratio_perf_maxdd = dict()
        self.max_dd = dict()
        self.vol = dict()
        self.sharpe_ratio = dict()

    def etudeBackTest(self):

        h = 5
        H = 260

        if list(self.y_bt.values())[0].index[-1] > dt.datetime.strptime('2018/07/30', '%Y/%m/%d'):

            ul = self.y_bt[self.underlying]
            rt_underlying = np.divide(ul, ul.shift(h)) - 1


            for k, bt in self.y_bt.items():
                # Volatilité
                rt = np.divide(bt, bt.shift(h)) - 1
                self.vol[k] = rt.rolling(H).std() * np.sqrt(H/h)
                self.correl[k] = rt.iloc[:, 0].rolling(window=H).corr(rt_underlying.iloc[:, 0])

                self.stat_global.loc['Vol', k] = (rt.std() * np.sqrt(H / h)).values
                self.stat_global.loc['Correlation', k] = rt.iloc[:, 0].corr(rt_underlying.iloc[:, 0])

                # Ratio de Sharpe
                self.perf[k] = np.divide(bt, bt.shift(H)) - 1
                self.sharpe_ratio[k] = np.divide(self.perf[k], self.vol[k])

                # Ratio de Sharpe Total
                self.stat_global.loc['Perf', k] = ((bt.iloc[-1] / bt.iloc[0])**(365/(bt.index[-1] - bt.index[0]).days)-1).values
                self.stat_global.loc['Sharpe', k] = self.stat_global.loc['Perf', k] / self.stat_global.loc['Vol', k]

                # Max drawdown
                if list(self.y_bt.values())[0].index[-1] > dt.datetime.strptime('2019/04/01', '%Y/%m/%d'):
                    rt = list()
                    for sh in range(1, 260):
                        rt.append(np.divide(bt, bt.shift(sh)) - 1)

                    df = pd.concat(rt, axis=1)
                    self.max_dd[k] = df.apply(lambda x: min(x), axis=1).rolling(H).min()

                    # Perf / Max DD
                    self.ratio_perf_maxdd[k] = np.divide(self.perf[k], -self.max_dd[k].to_frame())

                # Statistique sur les Poids
                self.stat_global.loc['Freq Long', k] = ((self.y_weight[k] > 0).sum() / self.y_weight[k].shape[0]).values
                self.stat_global.loc['Freq Short', k] = ((self.y_weight[k] < 0).sum() / self.y_weight[k].shape[0]).values
                self.stat_global.loc['Alternance de signe en nb de jours', k] = (self.y_weight[k].shape[0])/(1+(self.y_weight[k]*self.y_weight[k].shift(1) < 0).sum()).values

                self.stat_global.loc['Freq Prevision Positive', k] = ((self.y_pred[k] > 0).sum() / self.y_pred[k].shape[0]).values
                self.stat_global.loc['Freq Prevision Negative', k] = ((self.y_pred[k] < 0).sum() / self.y_pred[k].shape[0]).values
                self.stat_global.loc['Alternance position', k] = ((self.y_pred[k] * self.y_pred[k].shift(1) < 0).sum() / (self.y_pred[k].shape[0] - 1)).values


            # Version 1 du score de performance
            self.stat_global.loc['Score Mu', :] = self.stat_global.loc['Perf', :] - \
                                                  self.stat_global.loc['Correlation', :]\
                                                  * self.stat_global.loc['Perf', self.underlying]
            # Version 2 du score de performance
            self.stat_global.loc['Score Mu Risk', :] = self.stat_global.loc['Perf', :] - \
                                                       self.stat_global.loc['Correlation', :]\
                                                       * (self.stat_global.loc['Vol', :]/self.stat_global.loc['Vol', self.underlying])\
                                                       * self.stat_global.loc['Perf', self.underlying]

            # Version 3 du score de performance
            self.stat_global.loc['Score Mu Abs', :] = self.stat_global.loc['Perf', :] - \
                                                      np.abs(self.stat_global.loc['Correlation', :])\
                                                      * self.stat_global.loc['Perf', self.underlying]
            # Version 4 du score de performance
            self.stat_global.loc['Score Mu Risk Abs', :] = self.stat_global.loc['Perf', :] - \
                                                           np.abs(self.stat_global.loc['Correlation', :])\
                                                           * (self.stat_global.loc['Vol', :]/self.stat_global.loc['Vol', self.underlying])\
                                                           * self.stat_global.loc['Perf', self.underlying]


    def all_to_csv(self):

        # if list(self.y_bt.values())[0].index[-1] > dt.datetime.strptime('2017/12/31', '%Y/%m/%d'):

        # Extraction CSV des risques

        stat_global0 = self.ReadCsv(self.mypath + "\\" + self.ul_h + "--Stat_globales.csv")
        if stat_global0.__len__() > 0:
            self.stat_global = pd.concat([self.stat_global, stat_global0], axis=1, join='outer')

        self.stat_global.to_csv(self.mypath + "\\" + self.ul_h + "--Stat_globales.csv")

        df_perf_1y = self.to_csv(self.perf, "Perf 1Y")
        df_ratio_perf_maxdd = self.to_csv(self.ratio_perf_maxdd, "Ratio Perf 1Y MaxxDD")

        self.to_csv(self.max_dd, "MaxDD")
        self.to_csv(self.vol, "Vol")
        self.to_csv(self.correl, "Correlation")
        self.to_csv(self.sharpe_ratio, "Ratio de Sharpe")

        # Extraction CSV du backtest
        self.to_csv(self.y_bt, "BackTest")
        self.to_csv(self.y_pred, "Predictions")
        self.to_csv(self.y_weight, "Poids")

    def ReadCsv(self, path, InModel=False):

        if not os.path.isfile(path):
            return pd.DataFrame()

        stat = pd.read_csv(path, sep=",")

        try:
            stat.index = pd.to_datetime(stat.iloc[:, 0], format='%Y-%m-%d')
            stat.index.name = 'Date'
            stat = stat.drop([stat.columns[0]], axis=1)

        except ValueError:
            stat.index = stat.iloc[:, 0]
            stat.index.name = 'Index'
            stat = stat.drop([stat.columns[0]], axis=1)

        # Selection des colonnes des models que l'on cherche a estimer pour ne conservé que l'historique des
        # previsions de ces modèles
        if InModel:
            y_col = []
        else:
            y_col = list(stat.columns)

        for m in list(self.y_bt.keys()):
            if InModel:
                r = re.compile('.*' + m + '.*')
                y_col = y_col + list(filter(r.match, stat.columns))
            else:
                r = re.compile('^(?!' + m + ').*$')
                y_col = list(filter(r.match, y_col))

        stat = stat[y_col]

        # On ne retient que les lignes n'ayant pas de Nan
        NotnanIndex = stat.isna().sum(axis=1) == 0
        stat = stat.loc[NotnanIndex, :]

        if stat.empty:
            stat = pd.DataFrame()

        return stat

    def to_csv(self, data, nom_fichier):

        stat0 = self.ReadCsv(self.mypath + "\\" + self.ul_h + "--" + nom_fichier + ".csv")

        if data.__len__() > 0:
            df = Utilitaire.dict_list_to_df(data)
            if stat0.__len__()>0:
                df = pd.concat([df, stat0], axis=1, join='outer')
            df.to_csv(self.mypath + "\\" + self.ul_h + "--" + nom_fichier + ".csv")
            return df
        else:
            return None
