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

            self.Univers_Y = DataFrame
            self.Univers_X = DataFrame

            self.RawData = DataFrame
            self.Data = DataFrame

            self.X = DataFrame
            self.Y = DataFrame

            self.X_Out = DataFrame
            self.Y_Out = DataFrame

            self.X_BT = DataFrame
            self.Y_BT = DataFrame

            self.ListDataFrame0 = list()
            self.ListPrix = list()
            self.Data_Price = DataFrame

            self.ParametreDeDerivation = DataFrame
        else:
            A = pickle.load(open(os.path.abspath("").replace("Program_ML\\Code", "Data.pkl"), 'rb'))

            # Permet de ballayer les champs de l'object A et de les copier dans ceux de self qui sont les mêmes
            for k in A.__dict__.keys():
                setattr(self, k, getattr(A, k))

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
                if len(MasterD) != 0:
                    DerivP.append(read_csv(MasterD[0], sep=","))
                else:
                    print("Le fichier de dérivation Master n'existe pas!!")

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

                NewVariable = Serie.derivation(row['Operation'], row['KeepAtEnd'], H, P, List_E1, List_E2)

                if row['Operation'] in ['Multiplication', ''] \
                        or (isinstance(row['Operation'], float) and isnan(row['Operation'])):
                    for v in List_E1:
                        self.ListDataFrame0.remove(v)

                self.ListDataFrame0 = self.ListDataFrame0 + NewVariable

                elapseT.append(time.perf_counter() - t0)

        for v in self.ListDataFrame0:
            if v.S.shape[1] > 1:
                print("Le nombre de colonne de la serie " + v.Nom + " est supérieur à 1!!!")


        print("Temps de calcul total des dérivations : " + str(sum(elapseT)))

        # Filtration du prix des futures
        self.ListPrix = list(filter(lambda x: x.to_keep == 2, self.ListDataFrame0))
        self.Data_Price = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                                 list(map(lambda x: x.S, self.ListPrix)))

        # Filtration de la liste sur les series que l'on veut conserver
        self.ListDataFrame0 = list(filter(lambda x: x.to_keep == 1, self.ListDataFrame0))

        # Création du DataFrame global
        self.Data = pd.concat([x.S for x in self.ListDataFrame0], axis=1)
        # self.Data = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
        #                    list(map(lambda x: x.S, self.ListDataFrame0)))

        A = []
        for x in self.ListDataFrame0:
            di = x.__dict__
            di.pop('S', di)
            A.append(pd.DataFrame(di, index=[di['Nom']]))
        self.Univers_X = pd.concat(A, axis=0)
        self.Univers_X = self.Univers_X.drop(['Nom'], axis=1)


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
        kk = AbsoluteStart < self.Data_Price.index
        self.Data_Price = self.Data_Price.loc[kk, :]

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


    def listBT(self, path=os.path.abspath("").replace("\\Code", "\\Parametrage\\")):

        STR = read_csv(path + 'Strategie.csv', sep=",")

        st = []
        for index, row in STR.iterrows():

            col = list(row.index)[1:]

            if row[0] == 1:
                st0 = ""
                for c in col:
                    st0 = st0 + ":" if st0 != "" else st0
                    st0 = st0 + c + "=" + row.loc[c, ]

                st0 = [st0]
                #Dupplication des défintions selon les "ou /" imbriqué
                while len(list(filter(lambda x: x is not None,
                                      map(lambda x: re.search('\([_a-zA-Z0-9/]+\)', x), st0), ))) > 0:
                    for cond in st0:
                        # detection des conditions "ou" entre parentaises
                        orp = re.findall('\([_a-zA-Z0-9/]+\)', cond)
                        if len(orp) > 0:
                            st0.remove(cond)
                            C = orp[0]
                            Ck = re.sub('\(|\)', '', C).split('/')
                            for k in Ck:
                                st0.append(re.sub('\(' + C + '\)', k, cond))
                            break

                st = st + list(map(lambda x: re.split(':|=', x), st0))

        xo = []
        for p in st:
            p = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, p))

            zipList = zip([p[k] for k in range(0, len(p) - 1, 2)], [p[k] for k in range(1, len(p), 2)])
            p = dict(zipList)

            # Varaible à prevoir
            xk = list(filter(lambda x: x.Level1 == "Y", self.ListDataFrame0))

            if p['Inst'] is not None:
                xk = [x for x in xk if re.search(p['Inst'], x.Nom) is not None]

            if p['Horizon'] is not None:
                xk = [x for x in xk if x.h == p['Horizon']]

            if p['Y'] is not None:
                xk = [x for x in xk if x.DerivationName == p['Y']]

            #Prix de la variable à prévoir
            if p['Inst'] is not None:
                xp = [x for x in self.ListPrix if re.search(p['Inst'], x.Nom) is not None]

            xo = xo + [{'Yname': x.Nom, 'h': x.h, 'Algo': [p['Algo']], 'PrixInst': xp[0].Nom } for x in xk]
        return xo




    def listBT_II(self, path=os.path.abspath("").replace("\\Code", "\\Parametrage\\")):

        STR = read_csv(path + 'Strategie_II.csv', sep=",")

        st = []
        for index, row in STR.iterrows():
            col = list(row.index)
            st0 = ""
            for c in col:
                st0 = st0 + ":" if st0 != "" else st0

                if isinstance(row.loc[c, ], str):
                    st0 = st0 + c + "=" + row.loc[c, ]
                else:
                    st0 = st0 + c + "=" + str(row.loc[c,])

            st0 = [st0]
            #Dupplication des défintions selon les "ou /" imbriqué
            while len(list(filter(lambda x: x is not None,
                                  map(lambda x: re.search('\([_a-zA-Z0-9/]+\)', x), st0), ))) > 0:
                for cond in st0:
                    # detection des conditions "ou" entre parentaises
                    orp = re.findall('\([_a-zA-Z0-9/]+\)', cond)
                    if len(orp) > 0:
                        st0.remove(cond)
                        C = orp[0]
                        Ck = re.sub('\(|\)', '', C).split('/')
                        for k in Ck:
                            st0.append(re.sub('\(' + C + '\)', k, cond))
                        break

            st = st + list(map(lambda x: re.split(':|=', x), st0))

        Group = dict()
        for p in st:
            p = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, p))

            zipList = zip([p[k] for k in range(0, len(p) - 1, 2)], [p[k] for k in range(1, len(p), 2)])
            p = dict(zipList)

            if p['Groupe'] not in list(Group.keys()):
                Group[p['Groupe']] = []

            # Varaible à prevoir
            xk = list(filter(lambda x: x.Level1 == "Y", self.ListDataFrame0))

            if p['Inst'] is not None:
                xk = [x for x in xk if re.search(p['Inst'], x.Nom) is not None]

            if p['Horizon'] is not None:
                xk = [x for x in xk if x.h == p['Horizon']]

            if p['Y'] is not None:
                xk = [x for x in xk if x.DerivationName == p['Y']]

            #Prix de la variable à prévoir
            if p['Inst'] is not None:
                xp = [x for x in self.ListPrix if re.search(p['Inst'], x.Nom) is not None]

            Group[p['Groupe']] = Group[p['Groupe']] + \
                                 [{'Yname': x.Nom, 'h': x.h, 'PrixInst': xp[0].Nom,
                                 'Algo': p['Algo'], 'cv': p['cv'], 'strategie': p['strategie'],
                                 'refit': p['refit'], 'max_weight': p['max_weight'], 'Centrage': p['Centrage']} for x in xk]
        return Group

    def SelectData(self, Selected_Col):
        DataM0 = copy.copy(self)
        if Selected_Col is not None:
            # Données selectionnées

            for data in ['X', 'X_BT', 'X_Out']:
                setattr(DataM0, data, getattr(DataM0, data).loc[:, Selected_Col])

        return DataM0

    def All_bt(self, nom, strategies):
        for s in strategies:
            BT = BackTest(DataM=self,
                          y=s['Yname'],
                          price_name=s['PrixInst'],
                          strategie=s['strategie'],
                          list_model=s['Algo'],
                          refit=s['refit'],
                          h=s['h'],
                          cv=s['cv'],
                          mypath='res',
                          max_weight=s['max_weight'],
                          centrage=s['Centrage'])
            BT.estime_bt()
