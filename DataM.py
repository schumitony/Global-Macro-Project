from BackTest import BackTest
from pandas import read_csv, concat, to_datetime, DataFrame
from math import ceil, isnan
from Serie import Serie
import numpy as np

import time
import pickle
import glob
import os
import re
from functools import reduce
import pandas as pd
import copy
from dateutil.relativedelta import relativedelta
import datetime as dt
from LogFiles import log
from itertools import product


class DataM:

    def __init__(self, Load=False):
        if Load is False:
            self.Univers = DataFrame
            self.RawData = DataFrame
            self.Data = DataFrame

            self.X = DataFrame
            self.Y = DataFrame

            self.X_Out = DataFrame
            self.Y_Out = DataFrame

            self.X_BT = DataFrame
            self.Y_BT = DataFrame

            self.ListDataFrame0 = list()
            self.ListFuture = list()
            self.Future_Price = DataFrame

            self.ParametreDeDerivation = DataFrame
        else:
            A = pickle.load(open('Data.pkl', 'rb'))

            self.Univers = A.Univers
            self.RawData = A.RawData
            self.Data = A.Data

            self.X = A.X
            self.Y = A.Y

            self.X_Out = A.X_Out
            self.Y_Out = A.Y_Out

            self.X_BT = A.X_BT
            self.Y_BT = A.Y_BT

            self.ListDataFrame0 = A.ListDataFrame0
            self.ListFuture = A.ListFuture
            self.Future_Price = A.Future_Price
            self.ParametreDeDerivation = A.ParametreDeDerivation

    def copy(self):
        return copy.copy(self)

    def Derivation(self, path=os.path.abspath("").replace("\\Code", "\\Parametrage\\"), whichcase='All',
                   creatCSV=False):

        if whichcase == 'One':
            self.ParametreDeDerivation = read_csv(path + 'Derivation.csv', sep=",")
        elif whichcase == 'All':
            DerivP = list()

            MasterD = glob.glob(path + 'Master_Derivation.csv')
            if MasterD is not []:
                DerivP.append(read_csv(MasterD[0], sep=","))

            for files_name in glob.glob(path + 'Derivation_*.csv'):
                DerivP.append(read_csv(files_name, sep=","))

            self.ParametreDeDerivation = pd.concat(DerivP)

        elapseT = list()
        for index, row in self.ParametreDeDerivation.iterrows():
            t0 = time.perf_counter()

            # Filtration du premier groupe
            if isinstance(row['Groupe1'], str):
                List_E1 = self.ExtractList(row, 'Groupe1')

            # Filtration du deuxieme groupe
            if isinstance(row['Groupe2'], str):
                List_E2 = self.ExtractList(row, 'Groupe2')
            else:
                List_E2 = [None]

            # Recupération des Horizon de calculs
            if isinstance(row['Horizons'], str) or (isinstance(row['Horizons'], float) and not isnan(row['Horizons'])):
                H = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, row['Horizons'].split(";")))
            else:
                H = row['Horizons'] if not isnan(row['Horizons']) else [None]

            # Recupération des Paramètre de calculs
            if isinstance(row['Parametres'], str) or (isinstance(row['Parametres'], float) and not isnan(row['Parametres'])):
                P = self.ExtractList(row, 'Parametres')
            else:
                P = [row['Parametres']] if not isnan(row['Parametres']) else [None]

            if List_E1.__len__() == 0:
                print(
                    "Aucun instrument pour l'operation " + row['Operation'] + ": " + row['DerivationName'] + ": " + row[
                        'Groupe1'])
            else:

                NewVariable = Serie.derivation(row['Operation'], H, P, List_E1, List_E2)

                # Transformation des listes imbriquées en une liste simple de Series
                while isinstance(NewVariable[0], list) is True:
                    NewVariable = list(reduce(lambda y, x: y + x, filter(lambda x: isinstance(x, list) is True, NewVariable)))

                # Modification de le l'attribu to_keep et du nom de la colonne
                for v in NewVariable:
                    v.to_keep = row['KeepAtEnd']
                    v.S.columns = [v.Nom]

                if row['Operation'] in ['Multiplication']:
                    for v in List_E1:
                        self.ListDataFrame0.remove(v)


                self.ListDataFrame0 = self.ListDataFrame0 + NewVariable

                elapseT.append(time.perf_counter() - t0)

        for v in self.ListDataFrame0:
            if v.S.shape[1] > 1:
                print("Le nombre de colonne de la serie " + v.Nom + " est supérieur à 1!!!")

        # Filtration du prix des futures
        self.ListFuture = list(filter(lambda x: x.to_keep == 2, self.ListDataFrame0))
        self.Future_Price = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                                   list(map(lambda x: x.S, self.ListFuture)))

        # Filtration de la liste sur les series que l'on veut conserver
        self.ListDataFrame0 = list(filter(lambda x: x.to_keep == 1, self.ListDataFrame0))

        # Création du DataFrame global
        self.Data = pd.concat([x.S for x in self.ListDataFrame0], axis=1)
        # self.Data = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
        #                    list(map(lambda x: x.S, self.ListDataFrame0)))

        # Creation du fichier CSV avec toutes les series conservées
        if creatCSV is True:
            self.Data.to_csv(os.path.abspath("").replace("Program_ML\\Code", "AllDataDerivated.csv"))

        # Sauvgarde de l'object self avec pickle
        with open(os.path.abspath("").replace("Program_ML\\Code", "Data.pkl"), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def Decoupage(self, AbsoluteStart='2013/12/31', EndTest='2013/12/31', Bt_Duration=3, Test_Duration=6,
                  TypeDecoupage='Out'):

        dt_format = '%Y/%m/%d'

        if isinstance(AbsoluteStart, str) is False:
            AbsoluteStart = AbsoluteStart.strftime(dt_format)

        if isinstance(EndTest, str):
            end_test = dt.datetime.strptime(EndTest, dt_format)
        else:
            end_test = EndTest

        end_bt = end_test + relativedelta(months=Bt_Duration)
        EndBackTest = end_bt.strftime(dt_format)

        start_test = end_test + relativedelta(months=-Test_Duration)
        StartTest = start_test.strftime(dt_format)

        if TypeDecoupage == 'Out':
            EndTraining = StartTest
        elif TypeDecoupage == 'Overlap':
            EndTraining = end_test.strftime(dt_format)

        # Creation de la matrice des Raw X
        Col = [x.Nom for x in self.ListDataFrame0 if x.Level1 != "Y"]
        self.X = self.Data.loc[self.Data.index <= EndTraining, Col]

        kk = np.logical_and(StartTest < self.Data.index, self.Data.index <= EndTest)
        self.X_Out = self.Data.loc[kk, Col]

        kk = np.logical_and(EndTest < self.Data.index, self.Data.index <= EndBackTest)
        self.X_BT = self.Data.loc[kk, Col]

        # Creation de la matrice des Raw Y
        Col = [x.Nom for x in self.ListDataFrame0 if x.Level1 == "Y"]
        self.Y = self.Data.loc[self.Data.index <= EndTraining, Col]

        kk = np.logical_and(StartTest < self.Data.index, self.Data.index <= EndTest)
        self.Y_Out = self.Data.loc[kk, Col]

        kk = np.logical_and(EndTest < self.Data.index, self.Data.index <= EndBackTest)
        self.Y_BT = self.Data.loc[kk, Col]

        # Restriction des prix pour le backtest
        kk = AbsoluteStart < self.Future_Price.index
        self.Future_Price = self.Future_Price.loc[kk, :]

    def ExtractList(self, row, RowName):

        # Split des definitions différents ";"

        List_Car = row[RowName].split(';')
        # if isinstance(row[RowName], str):
        #     List_Car = row[RowName].split(';')
        # elif isinstance(row['Horizons'], float):
        #     List_Car = row[RowName]

        #Dupplication des défintions selon les "ou /" imbriqué
        while len(list(filter(lambda x: x is not None,
                              map(lambda x: re.search('\([a-zA-Z0-9/]+\)', x), List_Car), ))) > 0:
            for cond in List_Car:
                # detection des conditions "ou" entre parentaises
                orp = re.findall('\([a-zA-Z0-9/]+\)', cond)
                if len(orp) > 0:
                    List_Car.remove(cond)
                    C = orp[0]
                    Ck = re.sub('\(|\)', '', C).split('/')
                    for k in Ck:
                        List_Car.append(re.sub('\(' + C + '\)', k, cond))
                    break

        List_Car = list(map(lambda x: re.split(':|=', x), List_Car))

        if RowName in ['Groupe1', 'Groupe2']:
            # Filtration successive de nos series selon le nombre de niveaux definit dans le parametrage
            ii = 0
            List_F = list()
            while ii < List_Car.__len__():
                List_Car[ii] = list(map(lambda x: int(x) if x.isdigit() else x, List_Car[ii]))

                List_Arg = [List_Car[ii][k] for k in range(1, len(List_Car[ii]), 2)]
                List_Par = [List_Car[ii][k] for k in range(0, len(List_Car[ii])-1, 2)]

                List_E = list(filter(lambda x: self.MatchV(List_Par, List_Arg, row['DerivationName'], x),
                                     self.ListDataFrame0))

                List_F = List_F + List_E
                ii += 1
            return List_F
        else:
            return List_Car

    def MatchV(self, List_Car, List_Arg, Dev, x):
        boo = []
        for lcar, larg in zip(List_Car, List_Arg):
            boo.append(getattr(x, lcar) == larg)
        boo.append(x.DerivationName == Dev)

        return all(k is True for k in boo)

    def Loading(self, path=os.path.abspath("").replace("\\Code", "\\Latest data\\")):

        self.Univers = read_csv(path + 'Univers.csv', sep=",")

        # Q=read_csv(path + 'Equity_SP500_LAST.csv', sep=",")
        # Q.index=to_datetime(Q['Date'])
        # Q=Q.drop(['Date'], axis=1)

        for files_name in glob.glob(path + '*.csv'):
            name = files_name.replace(path, "").replace(".csv", "")

            if name != 'Univers' and self.Univers["InOut"][self.Univers["Nom"] == name].iloc[0] >= 1:
                try:
                    Q = Serie()
                    Q.Loading(files_name, path, self.Univers)
                    if (Q.S.count(0) != 0).any():
                        self.ListDataFrame0.append(Q)

                except Exception as Ex:
                    res_file = log(path=os.path.abspath("").replace("\\Code", "\\Log\\"), nom="Erreurs_chargement",
                                   create=True)
                    res_file.write('Impossible de charger le fichier %s ' % (name))

    def ClearData(self, creatCSV=False):

        self.RawData = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                              list(map(lambda x: x.S, self.ListDataFrame0)))

        # Suppression des historiques anterieurs à dec 1998
        self.RawData = self.RawData.loc[self.RawData.index > '1997/12/31', :]

        # Calcul de la proportion de données manquante par date
        nb_nan = self.RawData.isna().sum(axis=1) / self.RawData.shape[1]

        # Suppression des dates ayant une proportion de Nan trop elevée
        self.Data = self.RawData.loc[nb_nan < 0.5, :]
        self.RawData = self.RawData.loc[nb_nan < 0.5, :]

        # Cristalisation des données manquantes
        self.Data = self.Data.fillna(method='ffill')

        # Calcul de la proportion de données manquante par serie
        nb_nan = self.RawData.isna().sum(axis=0) / self.RawData.shape[0]

        # Suppression des series ayant une proportion de Nan trop elevee au debut
        Col = self.RawData.columns[nb_nan < 0.5]
        self.ListDataFrame0 = [x for x in self.ListDataFrame0 if x.Nom in Col]

        self.Data = self.Data.loc[:, Col]
        self.RawData = self.RawData.loc[:, Col]

        # Certaines series sont constantes sur des temps assez long.... en raison de Nan cristaliser ou de données mal contribuées
        t0 = time.perf_counter()
        DaysC = self.NbDayConst()
        t1 = time.perf_counter() - t0

        # self.Data.to_csv(os.path.abspath("").replace("\\Code", "\\AllData_Raw.csv"))
        #
        # nbDays = (self.Data - self.Data.shift(1) == 0).rolling(30).sum()
        # nbDaysII = np.divide((nbDays > 20).sum(), nbDays.shape[0])

        # Col_Excl = DaysC.columns[(DaysC.loc['rXtrem80_Max', :] > 40) | (DaysC.loc['rXtrem95_Max', :] > 30)]
        Col_Excl = DaysC.columns[(DaysC.loc['rXtrem80_Max', :] > 40)]
        self.Data.loc[:, Col_Excl].to_csv(os.path.abspath("").replace("Program_ML\\Code", "AllData_Excl.csv"))

        # Col = DaysC.columns[(DaysC.loc['rXtrem80_Max', :] <= 40) & (DaysC.loc['rXtrem95_Max', :] <= 30)]
        Col = DaysC.columns[(DaysC.loc['rXtrem80_Max', :] <= 40)]
        self.ListDataFrame0 = [x for x in self.ListDataFrame0 if x.Nom in Col]

        self.Data = self.Data.loc[:, Col]
        self.RawData = self.RawData.loc[:, Col]

        if creatCSV is True:
            # Creation du CSV Global syncronisé
            self.Data.to_csv(os.path.abspath("").replace("Program_ML\\Code", "AllData.csv"))

        # Remplacement des données d'origines par les données nettoyées
        for ncol in self.Data:
            QQ = list(filter(lambda x: x.Nom == ncol, self.ListDataFrame0))[0]
            QQ.S = self.Data.loc[:, ncol].to_frame(name=ncol)

            # Frequence de la serie
            QQ.Freqence = (QQ.S.index[1:] - QQ.S.index[0:-1]).days.astype(float).values.mean()

        # Suppression des variables n'ayant pas toutes les données complétées
        # self.ListDataFrame0 = [x for x in self.ListDataFrame0 if x.S.isnan().values.any() is False]

    # def NbDayConst(self):
    #     for j in range(0, self.Data.shape[1]):
    #         for i in range(0, self.Data.shape[0]):
    #           if i == 0:
    #               DD[i, j] = ConstDays.iloc[i, j]
    #                   else:
    #               DD[i, j] = (DD[i-1, j] + ConstDays.iloc[i, j]) * ConstDays.iloc[i, j]

    def NbDayConst(self):
        ConstDays = self.Data - self.Data.shift(1) == 0
        DD = np.zeros(ConstDays.shape)
        DD_I = np.empty(ConstDays.shape)
        DD_I[:] = np.NaN

        i = 0
        for line in ConstDays.itertuples(index=True):
            if line[0] == ConstDays.index[0]:
                DD[i, :] = line[1:]
            else:
                DD[i, :] = (DD[i - 1, :] + line[1:]) * line[1:]

            if i > 1:
                # k = (DD[i, :] == 0) & (DD[i-1, :] != 0)
                # DD_I[i, k] = DD[i-1, k]
                k = (DD[i, :] == 0)
                DD_I[i - 1, k] = DD[i - 1, k] + 1
            i = i + 1

        DD_I[- 1, :] = DD[- 1, :] + 1

        Qtl = np.nanquantile(DD_I, q=[0.5, 0.8, 0.95, 1], axis=0)
        DaysC = pd.DataFrame(Qtl, ['Q50', 'Q80', 'Q95', 'Max'], self.Data.columns)

        # Aux = nbDaysConst.loc['Q80', :] - nbDaysConst.loc['Q50', :]
        # Aux[Aux <= 1] = 1
        # nbDaysConst.loc['rXtrem80_50', :] = np.divide(nbDaysConst.loc['Max', :] - nbDaysConst.loc['Q80', :], Aux)
        #
        # Aux = nbDaysConst.loc['Q95', :] - nbDaysConst.loc['Q50', :]
        # Aux[Aux <= 1] = 1
        # nbDaysConst.loc['rXtrem95_50', :] = np.divide(nbDaysConst.loc['Max', :] - nbDaysConst.loc['Q95', :], Aux)

        DaysC.loc['rXtrem80_Max', :] = np.divide(DaysC.loc['Max', :], DaysC.loc['Q80', :])
        DaysC.loc['rXtrem95_Max', :] = np.divide(DaysC.loc['Max', :], DaysC.loc['Q95', :])

        DaysC.transpose().to_csv(os.path.abspath("").replace("Program_ML\\Code", "DataConst.csv"))
        # pd.DataFrame(DD_I, self.Data.index, self.Data.columns).to_csv(os.path.abspath("").replace("\\Code", "\\DataNbConst.csv"))

        return DaysC

    def listBT(self, fut=[None], horizon=[None], deriv=[None]):
        # Liste des Y

        # On peut filtrer la liste des futures avec la variable fut
        xo = []
        for f in fut:
            for h in horizon:
                for d in deriv:

                    xk = [x for x in self.ListDataFrame0 if x.Level1 == "Y"]

                    if f is not None:
                        xk = [x for x in xk if re.search(f, x.Nom) is not None]

                    if h is not None:
                        xk = [x for x in xk if x.h == h]

                    if d is not None:
                        xk = [x for x in xk if x.DerivationName == d]

                    xo = xo + [(x.Nom, x.h) for x in xk]

                    # if f is None and h is None:
                    #     xo = xo + [(x.Nom, x.h) for x in self.ListDataFrame0 if x.Level1 == "Y" and x.DerivationName == d]
                    # elif f is not None and h is None:
                    #     xo = xo + [(x.Nom, x.h) for x in self.ListDataFrame0 if x.Level1 == "Y" and x.DerivationName == d
                    #                and re.search(f, x.Nom) is not None]
                    # elif f is None and h is not None:
                    #     xo = xo + [(x.Nom, x.h) for x in self.ListDataFrame0 if x.Level1 == "Y" and x.DerivationName == d
                    #                and x.h == h]
                    # elif f is not None and h is not None:
                    #     xo = xo + [(x.Nom, x.h) for x in self.ListDataFrame0 if x.Level1 == "Y" and x.DerivationName == d
                    #                and re.search(f, x.Nom) is not None and x.h == h]

        y, h = zip(*xo)

        # y = [x.Nom for x in self.ListDataFrame0 if x.Level1 == "Y"]
        # h = [x.h for x in self.ListDataFrame0 if x.Level1 == "Y"]

        l = list()
        for y0 in self.ListFuture:

            # Liste des futurs
            r = re.compile(".*" + y0.Nom)
            y_list = list(filter(r.match, y))

            for y00 in y_list:
                l.append((y00, y0.Nom))
        return l, h

    def SelectData(self, Selected_Col):
        DataM0 = copy.copy(self)
        if Selected_Col is not None:
            # Données selectionnées

            for data in ['X', 'X_BT', 'X_Out']:
                setattr(DataM0, data, getattr(DataM0, data).loc[:, Selected_Col])

        return DataM0

    def All_bt(self, list_bt, h, cv, strategie, refit, list_model, max_weight):
        for y0, h0 in zip(list_bt, h):
            BT = BackTest(DataM=self,
                          y=y0[0],
                          price_name=y0[1],
                          strategie=strategie,
                          list_model=list_model,
                          refit=refit,
                          h=h0,
                          cv=cv,
                          mypath='res',
                          max_weight=max_weight)

            BT.estime_bt()
