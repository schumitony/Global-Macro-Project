from pandas import read_csv, concat, to_datetime, DataFrame
from math import ceil, isnan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import time


class Serie:
    def __init__(self):
        self.S = DataFrame
        self.Freqence = float

        self.to_keep = 0
        self.h = 0
        self.sum_h = 0
        self.Nom = ""
        self.DerivationName = ""
        self.Level1 = ""
        self.Level2 = ""
        self.Level3 = ""
        self.Level4 = ""
        self.Pays = ""
        self.Maturity = ""
        self.Lag = float
        self.DerivationLevel = float

    def CopyCar(self, h=0):
        Q = Serie()

        Q.h = h
        Q.sum_h = self.sum_h + h

        Q.Freqence = self.Freqence

        Q.Nom = self.Nom
        Q.Level1 = self.Level1
        Q.Level2 = self.Level2
        Q.Level3 = self.Level3
        Q.Level4 = self.Level4
        Q.Pays = self.Pays
        Q.Maturity = self.Maturity
        Q.Lag = self.Lag
        Q.DerivationName = self.DerivationName
        Q.DerivationLevel = self.DerivationLevel

        return Q

    def Loading(self, files_name, path, Univers):

        self.Nom = files_name.replace(path, '').replace('.csv', '')

        # Referencement de la serie
        self.Level1 = Univers[Univers['Nom'] == self.Nom]['Level1'].values[0]
        self.Level2 = Univers[Univers['Nom'] == self.Nom]['Level2'].values[0]
        self.Level3 = Univers[Univers['Nom'] == self.Nom]['Level3'].values[0]
        self.Level4 = Univers[Univers['Nom'] == self.Nom]['Level4'].values[0]
        self.Maturity = Univers[Univers['Nom'] == self.Nom]['Maturity'].values[0]
        self.Pays = Univers[Univers['Nom'] == self.Nom]['Pays'].values[0]
        self.Lag = Univers[Univers['Nom'] == self.Nom]['Lag'].values[0]
        self.DerivationLevel = 0

        self.S = read_csv(files_name, sep=",")
        self.S.index = to_datetime(self.S['Date'], format='%Y-%m-%d')
        self.S = self.S.drop(['Date'], axis=1)

        # if Univers["InOut"][Univers["Nom"] == self.Nom].iloc[0] == 2:
        #     self.to_keep = 2

        if "Return" in self.Nom:
            self.Level4 = "Last"
            self.Nom = self.Nom.replace('_Return', '')

            # to_keep = 2 pour garder le prix des futures en fin de traitement!
            # self.to_keep = 2

            # Suppresion des champs autre que Return
            list_to_drop = [x for x in self.S.columns if x != "Valeur"]
            self.S = self.S.drop(list_to_drop, axis=1)

            # Nettoyage self.S=='Non Numérique'
            if self.S["Valeur"].dtype != 'float64':
                self.S = self.S[self.S["Valeur"] != 'Non Numérique']
                self.S = self.S.astype({"Valeur": 'float'})

            # Return + 1
            self.S = self.S + 1

            # Ajout d'une ligne avec 100!
            d = to_datetime(self.S.index[0] + pd.DateOffset(days=-1))
            self.S = pd.concat(
                [pd.DataFrame(data=[100], index=pd.date_range(start=d, end=d), columns=self.S.columns), self.S], axis=0)

            # Cummule des returns quotidiens pour calculer un prix fictif
            self.S = self.S.cumprod(axis=0)

        self.DerivationName = "Raw"

        self.S.columns = [self.Nom]

        # Frequence de la serie
        self.Freqence = (self.S.index[1:] - self.S.index[0:-1]).days.astype(float).values.mean()

        # Application du lag de la donnée
        if self.Lag != 0:
            self.S = self.S.shift(self.Lag)

    @staticmethod
    def derivation(TypeDerivation, Keep, ListH=None, ListP=None, ListeS1=None, ListeS2=None):
        listNewE = list()

        # if ListeS2 != [None]:
        #     List_E12 = list()
        #     [List_E12.append((x, y)) for x in ListeS1 for y in ListeS2 if x.Nom != y.Nom and (y, x) not in List_E12]

        if (isinstance(TypeDerivation, str) and TypeDerivation != "") \
                or (isinstance(TypeDerivation, float) and not isnan(TypeDerivation)):
            for p in ListP:
                if isinstance(p, list):
                    p = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, p))

                    zipList = zip([p[k] for k in range(0, len(p) - 1, 2)], [p[k] for k in range(1, len(p), 2)])
                    p = dict(zipList)

                if ListeS2 != [None]:
                    for h in ListH:
                        h = h if ListH != [None] else None
                        # listNewE.append(list(map(lambda x: getattr(x, TypeDerivation)(h, p, ListeS2), ListeS1)))
                        for el in ListeS1:
                            ListeS2 = list(filter(lambda x: x.Nom != el.Nom, ListeS2))
                            listNewE.append(getattr(el, TypeDerivation)(h, p, ListeS2))

                elif TypeDerivation in ['EcartMedGroupe']:
                    for h in ListH:
                        listNewE.append(getattr(Serie, TypeDerivation)(ListeS1, h, p))

                else:
                    for h in ListH:
                        h = h if ListH != [None] else None
                        listNewE.append(list(map(lambda x: getattr(x, TypeDerivation)(h, p), ListeS1)))

            # Transformation des listes imbriquées en une liste simple de Series
            while isinstance(listNewE[0], list) is True:
                listNewE = list(reduce(lambda y, x: y + x, filter(lambda x: isinstance(x, list) is True, listNewE)))
        else:
            listNewE = ListeS1

        # Modification de le l'attribu to_keep et du nom de la colonne
        for v in listNewE:
            v.to_keep = Keep
            v.S.columns = [v.Nom]

        return listNewE

    def DiffAbsolue(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'DAbs_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Diff"
        Q.DerivationLevel = Q.DerivationLevel + 1

        # Q.S.drop(Q.S.index, inplace=True)
        # Q.S = DataFrame(self.S.values[h:] - self.S.values[0:-h], index=self.S.index[h:], columns=['Valeur'])
        # Q.S.columns = self.S.columns
        Q.S = np.subtract(self.S, self.S.shift(h))

        return Q

    def DiffRelative(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'DRel_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Diff"
        Q.DerivationLevel = Q.DerivationLevel + 1

        Q.S = np.divide(self.S, self.S.shift(h)).apply(lambda x: x - 1)
        # Q.S = DataFrame(self.S.values[h:] / self.S.values[0:-h] - 1, index=self.S.index[h:], columns=['Valeur'])
        # Q.S.columns = self.S.columns

        return Q

    def Multiplication(self, h0=None, p=None):
        self.S = self.S * p['p_mult']
        return self

    def EcartQuantile(self, h0, p=None, Q1=None):

        Lq = list()

        h = ceil(h0 / self.Freqence)

        # Calcul des quantiles souhaité en décalant les résultats d'un jour (exclusion du dernier jour)
        Q0 = self.S.rolling(h).quantile(0.05)
        Q1 = self.S.rolling(h).quantile(0.25)
        Q2 = self.S.rolling(h).quantile(0.5)
        Q3 = self.S.rolling(h).quantile(0.75)
        Q4 = self.S.rolling(h).quantile(0.95)

        A = Q1.sub(Q0)
        B = Q2.sub(Q1)
        C = Q3.sub(Q2)
        D = Q4.sub(Q3)

        # Ecart des quantiles 0.95 et 0.05
        Q = self.CopyCar(h0)
        Q.Nom = 'Quantile_0.05_Vs_0.95_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Quantile"
        Q.DerivationLevel = Q.DerivationLevel + 1

        Q.S = (A + B + C + D)
        Lq.append(Q)

        # Ratio des quantiles 0.95 et 0.05
        Q = self.CopyCar(h0)
        Q.Nom = 'Ratio_Quantile_0.05_Vs_0.95_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Quantile"
        Q.DerivationLevel = Q.DerivationLevel + 1

        Q.S = (A + B).divide(C + D)
        Lq.append(Q)

        return Lq

    @staticmethod
    def EcartMedGroupe(ListeS, h=None, p=None):

        if p is not None:
            KeyList = list(p.keys())
            if 'p_h' in KeyList:
                ListeS = list(filter(lambda x: x.h == p['p_h'], ListeS))
            if 'p_sum_h' in KeyList:
                ListeS = list(filter(lambda x: x.sum_h == p['p_sum_h'], ListeS))
            if 'p_level4' in KeyList:
                ListeS = list(filter(lambda x: x.Level4 == p['p_level4'], ListeS))

        AllSeries = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                           list(map(lambda x: x.S, ListeS)))

        Median = AllSeries.median(axis=1)
        MedianExcess = AllSeries.subtract(Median, axis=0)

        ListE = list()
        for Sf in ListeS:
            S = Sf.CopyCar()
            S.Nom = 'EcartMedGroupe_' + S.Nom
            S.DerivationName = S.DerivationName + "_EcartMedGroupe"
            S.DerivationLevel = S.DerivationLevel + 1
            S.S = MedianExcess.loc[:, Sf.Nom].to_frame()
            ListE.append(S)

        return ListE

    def Autocorrelation(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'AutoCorrel_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_AutoCorrel"
        Q.DerivationLevel = Q.DerivationLevel + 1

        kk = pd.merge(self.S, self.S.shift(h), left_index=True, right_index=True)
        # Calcul des volatilité souhaité en decalant les resultats d'un jour (exclusion du dernier jour)
        Q.S = kk.iloc[:, 0].rolling(h).corr(kk.iloc[:, 1]).to_frame()

        return Q

    def Volatilite(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'Vol_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Vol"
        Q.DerivationLevel = Q.DerivationLevel + 1

        # Calcul des volatilité souhaité en decalant les resultats d'un jour (exclusion du dernier jour)
        # Q.S = self.S.rolling(h).std().shift(1)
        Q.S = self.S.rolling(h).std().apply(lambda x: x * np.sqrt(365 / self.h))

        return Q

    def Correlation(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'Correlation_' + str(h0) + 'd_' + Q.Nom + '_Vs_' + Q1.Nom
        Q.DerivationName = Q.DerivationName + "_Correlation"
        Q.DerivationLevel = Q.DerivationLevel + 1

        # Fusion des deux series pour les synchroniser
        Qk = pd.merge(Q1.S, self.S, left_index=True, right_index=True)

        # Calcul des correlation souhaité
        Q.S = Qk.iloc[:, 0].rolling(h).corr(Qk.iloc[:, 1]).to_frame()

        return Q

    def Spread(self, h0=None, p=None, ListeS=None):

        if p is not None:
            KeyList = list(p.keys())
            if 'p_same_country' in KeyList and p['p_same_country'] == 'Y':
                ListeS = list(filter(lambda x: x.Pays == self.Pays, ListeS))

            if 'p_different_country' in KeyList and p['p_different_country'] == 'Y':
                ListeS = list(filter(lambda x: x.Pays != self.Pays, ListeS))

            if 'p_same_h' in KeyList and p['p_same_h'] == 'Y':
                ListeS = list(filter(lambda x: x.Nom != self.Nom and x.h == self.h, ListeS))

        ListE = list()
        if len(ListeS) > 0:

            AllSeries = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'),
                               list(map(lambda x: x.S, ListeS)))

            Spread = AllSeries.subtract(self.S.iloc[:, 0], axis=0) * -1

            for Sf in ListeS:

                S = Sf.CopyCar()
                if 'Level4' in KeyList and p['Level4'] in ['SpreadGov', 'TxReel']:
                    S.Nom = p['Level4'] + '_' + self.Maturity + '_' + self.Pays

                elif 'Level4' in KeyList and p['Level4'] == 'Pente':
                    S.Nom = p['Level4'] + '_' + self.Level2 + '_' + S.Maturity + '_' + self.Maturity + '_' + self.Pays

                else:
                    S.Nom = 'Spread_' + self.Nom + '_Vs_' + S.Nom

                if S.Pays != self.Pays:
                    S.Pays = self.Pays + ' vs ' + S.Pays

                if isinstance(S.Maturity, str) and S.Maturity != self.Maturity:
                    S.Maturity = self.Maturity + ' vs ' + S.Maturity

                S.Level1 = self.Level1
                S.Level2 = self.Level2
                S.Level3 = self.Level3

                if p is not None and 'Level4' in KeyList:
                    S.Level4 = p['Level4']
                    if p['Level4'] == 'SpreadGov':
                        S.Pays = self.Pays

                S.DerivationName = S.DerivationName + "_Spread"
                S.DerivationLevel = S.DerivationLevel + 1
                S.S = Spread.loc[:, Sf.Nom].to_frame()

                ListE.append(S)

        return ListE

    def TempsExtrema(self, h0=None, p=None, ListeS=None):
        h = ceil(h0 / self.Freqence)

        idxDate = self.S.index[h - 1:].values

        # Temps écoulé depuis le dernier Max
        Qmax = self.CopyCar(h0)

        Qmax.Nom = 'TimeFromMax_' + str(h0) + 'd_' + Qmax.Nom
        Qmax.DerivationName = Qmax.DerivationName + "_TimeFromMax"
        Qmax.DerivationLevel = Qmax.DerivationLevel + 1
        Qmax.Level4 = "TimeFromMax"

        # Temps écoulé depuis le dernier Min
        Qmin = self.CopyCar(h0)

        Qmin.Nom = 'TimeFromMin_' + str(h0) + 'd_' + Qmin.Nom
        Qmin.DerivationName = Qmin.DerivationName + "_TimeFromMin"
        Qmin.DerivationLevel = Qmin.DerivationLevel + 1
        Qmin.Level4 = "TimeFromMin"

        # Perf depuis le dernier Max
        Pmax = self.CopyCar(h0)

        Pmax.Nom = 'PerfFromMax_' + str(h0) + 'd_' + Pmax.Nom
        Pmax.DerivationName = Pmax.DerivationName + "_PerfFromMax"
        Pmax.DerivationLevel = Pmax.DerivationLevel + 1
        Pmax.Level4 = "PerfFromMax"

        # Perf depuis le dernier Min
        Pmin = self.CopyCar(h0)

        Pmin.Nom = 'PerfFromMin_' + str(h0) + 'd_' + Pmin.Nom
        Pmin.DerivationName = Pmin.DerivationName + "_PerfFromMin"
        Pmin.DerivationLevel = Pmin.DerivationLevel + 1
        Pmin.Level4 = "PerfFromMin"

        # La ligne ci dessous calcul l'ArgMax sur la fenetre glissante de taille H. Si le resulat est H, le max se trouve sur cette ligne
        # elapseT = list()
        # t0 = time.perf_counter()

        i = 0

        PerfFromMax = []
        PerfFromMin = []

        TimeFromMax = []
        ArgMax = 0
        Max = -1

        TimeFromMin = []
        ArgMin = 0
        Min = 100000000
        for line in self.S.itertuples(index=True):
            # Recherche du Max
            if TimeFromMax == [] or line[1] > Max:
                ArgMax = line[0]
                Max = line[1]
            elif (line[0] - ArgMax).days > h0:
                a = self.S.iloc[max(0, i - h + 1):i + 1, :]
                ArgMax = a.index[np.argmax(a)]
                Max = np.max(a).values[0]

            # Recherche du Min
            if TimeFromMin == [] or line[1] < Min:
                ArgMin = line[0]
                Min = line[1]
            elif (line[0] - ArgMin).days > h0:
                a = self.S.iloc[max(0, i - h + 1):i + 1, :]
                ArgMin = a.index[np.argmin(a)]
                Min = np.min(a).values[0]

            i += 1
            TimeFromMax.append((line[0] - ArgMax).days)
            TimeFromMin.append((line[0] - ArgMin).days)

            PerfFromMax.append(line[1] / Max - 1)
            PerfFromMin.append(line[1] / Min - 1)

        Qmax.S = pd.DataFrame(TimeFromMax, index=self.S.index, columns=['TimeToMax'])
        Qmin.S = pd.DataFrame(TimeFromMin, index=self.S.index, columns=['TimeToMin'])

        Pmax.S = pd.DataFrame(PerfFromMax, index=self.S.index, columns=['PerfToMax'])
        Pmin.S = pd.DataFrame(PerfFromMin, index=self.S.index, columns=['PerfToMin'])

        # elapseT.append(time.perf_counter() - t0)
        # t0 = time.perf_counter()
        #
        # ArgMax = (self.S.rolling(h).apply(lambda x: np.argmax(x)) - (h - 1)).values[:, 0]
        #
        # idx = (np.arange(0, len(ArgMax)) + ArgMax)
        #
        # idx = idx[np.isnan(idx) == 0].astype(np.int)
        # ArgMax = self.S.index[idx]
        #
        # Qmax.S0 = pd.DataFrame(data=(idxDate - ArgMax).astype('timedelta64[D]'), index=idxDate)
        #
        # # Temps écoulé depuis le dernier Min
        # ArgMin = (self.S.rolling(h).apply(lambda x: np.argmin(x)) - (h-1)).values[:, 0]
        # idx = (np.arange(0, len(ArgMin)) + ArgMin)
        #
        # idx = idx[np.isnan(idx) == 0].astype(np.int)
        # ArgMin = self.S.index[idx]
        #
        # Qmin.S0 = pd.DataFrame((idxDate - ArgMin).astype('timedelta64[D]'), index=idxDate, columns=list(self.S.columns))
        #
        # elapseT.append(time.perf_counter() - t0)

        # ArgExtrema = np.array(list(map(lambda x, y: max(x, y), ArgMax, ArgMin)))
        #
        # ArgExtrema = np.array(list(map(lambda x: x.to_datetime64(), list(ArgExtrema))))
        #
        # dd = np.empty(idxDate.shape[0])
        #
        # LastIsMax = ArgExtrema == ArgMax
        # dd[LastIsMax] = (ArgExtrema[LastIsMax] - idxDate[LastIsMax]).astype('timedelta64[D]')

        return [Qmax, Qmin, Pmax, Pmin]

    def Zscore(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'Zscore_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Zscore"
        Q.DerivationLevel = Q.DerivationLevel + 1

        # Calcul des volatilité souhaité en decalant les resultats d'un jour (exclusion du dernier jour)
        Sigma = self.S.rolling(h).std()
        # On fixe les zero de la serie Sigma au minimum de la Serie des valeures strictement positive!
        m = np.min(Sigma[Sigma > 0]).values[0]
        Sigma = Sigma.applymap(lambda x: x if x != 0 else m)

        Mu = self.S.rolling(h).mean()

        Q.S = np.divide(np.subtract(self.S, Mu), Sigma)

        return Q

    def PositiveReturn(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'PositiveReturn' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_PositiveReturn"
        Q.DerivationLevel = Q.DerivationLevel + 1
        Q.Level1 = "Y"
        Q.Level4 = "PositiveReturn"

        # Q.S = DataFrame(self.S.values[h:] / self.S.values[0:-h] - 1, index=self.S.index[h:], columns=['Valeur'])

        # Dalage des donnée entre 1 jour et h jour
        Num = pd.concat([self.S.shift(-h) for h in list(range(1, h + 1))], axis=1)
        Denom = pd.concat([self.S for h in list(range(1, h + 1))], axis=1)

        # Calcul des perfs sur les horizon compris entre 1 et h
        Quotient = np.divide(Num, Denom).applymap(lambda x: x - 1)

        # Suppréssion des calculs aux dates n'ayant pas les h horizon de comptet!
        ii = np.any(np.isnan(Quotient), axis=1)

        # Calcul du nombre de perf positive sur les h perf possibles
        Q.S = np.sum(Quotient > 0, axis=1).map(lambda x: x / h).to_frame()

        Q.S.loc[ii] = np.nan

        return Q

    def ReturnClass(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'ReturnClass_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_ReturnClass"
        Q.DerivationLevel = Q.DerivationLevel + 1
        Q.Level1 = "Y"
        Q.Level4 = "ReturnClass"

        ret0 = np.divide(self.S.shift(-h), self.S).applymap(lambda x: x - 1)

        cla = np.empty((len(ret0), 1))
        cla.fill(np.nan)
        lis = []
        hp = ceil(p['p_h'] / self.Freqence)

        for wk in [-1, 1]:
            if wk == 1:
                ret = ret0[ret0 > 0]
                w = 0
            elif wk == -1:
                ret = ret0[ret0 <= 0]
                w = -2

            for k in [0.05, 0.95, 1]:
                lo = ret.dropna().rolling(pd.offsets.Day(p['p_h'])).quantile(k)
                A = pd.concat([lo, ret], axis=1).iloc[:, 0].to_frame()

                # lo = ret.rolling(hp).apply(lambda x: x.dropna().quantile(k))
                cla[np.logical_and(np.array(ret <= A), np.isnan(cla))] = w
                w += 1
                lo.columns = ['Q' + str(k)]
                lis.append(lo)

        cla[0:hp] = np.nan
        Q.S = pd.DataFrame(cla, index=self.S.index)

        return Q

    def Return(self, h0, p=None, Q1=None):
        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'Return_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Return"
        Q.DerivationLevel = Q.DerivationLevel + 1
        Q.Level1 = "Y"
        Q.Level4 = "Return"

        # Q.S = DataFrame(self.S.values[h:] / self.S.values[0:-h] - 1, index=self.S.index[h:], columns=['Valeur'])
        Q.S = np.divide(self.S.shift(-h), self.S).applymap(lambda x: x - 1)

        return Q

    def Normalized_Return(self, h0, p=None, Q1=None):

        h = ceil(h0 / self.Freqence)
        rt0 = np.divide(self.S, self.S.shift(h)) - 1
        Sigma = rt0.rolling(90).std()

        h = ceil(h0 / self.Freqence)
        Q = self.CopyCar(h0)

        Q.Nom = 'Normalized_Return_' + str(h0) + 'd_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Normalized_Return"
        Q.DerivationLevel = Q.DerivationLevel + 1
        Q.Level1 = "Y"
        Q.Level4 = "NormalizedReturn"

        Q.S = np.divide(np.divide(self.S.shift(-h), self.S).applymap(lambda x: x - 1), Sigma)

        return Q

    def Xtime(self, h0=None, p=None, Q1=None):

        h0 = 7
        h = ceil(h0 / self.Freqence)
        rt0 = np.divide(self.S, self.S.shift(h)) - 1
        Sigma_h0 = rt0.rolling(90).std() * p

        rt = list()
        delta_max = 260
        for sh in range(1, delta_max):
            rt.append(np.divide(self.S.shift(-sh), self.S) - 1)
            rt[-1].columns = [sh]

        df = pd.concat(rt, axis=1)

        nbj = list()
        nbj_pos = list()
        nbj_neg = list()

        for d in df.index:
            if not isnan(Sigma_h0.loc[d].values):

                ll = np.nonzero(df.loc[d, :].values > Sigma_h0.loc[d].values)
                if ll[0].__len__() > 0:
                    nbj_pos.append(ll[0][0] + 1)
                else:
                    nbj_pos.append(delta_max)

                ll = np.nonzero(df.loc[d, :].values < - Sigma_h0.loc[d].values)
                if ll[0].__len__() > 0:
                    nbj_neg.append(ll[0][0] + 1)
                else:
                    nbj_neg.append(delta_max)

                if nbj_pos[-1] < nbj_neg[-1]:
                    nbj.append(nbj_pos[-1])
                else:
                    nbj.append(-nbj_neg[-1])
            else:
                nbj.append(float('Nan'))
                nbj_pos.append(float('Nan'))
                nbj_neg.append(float('Nan'))

        nbj = pd.DataFrame(nbj, index=df.index)
        nbj_pos = pd.DataFrame(nbj_pos, index=df.index)
        nbj_neg = pd.DataFrame(nbj_neg, index=df.index)

        nbj_pos.quantile(np.arange(0, 1, 0.1))
        nbj_neg.quantile(np.arange(0, 1, 0.1))

        # import os
        # nbj_pos.to_csv(os.path.abspath("").replace("\\Code", "\\NB_Jours_Positif".csv"))
        # nbj_neg.to_csv(os.path.abspath("").replace("\\Code", "\\NB_Jours_Negatif.csv"))
        # nbj.to_csv(os.path.abspath("").replace("\\Code", "\\NB_Jours.csv"))
        # df.to_csv(os.path.abspath("").replace("\\Code", "\\Perf.csv"))
        # Sigma_h0.to_csv(os.path.abspath("").replace("\\Code", "\\Sigma.csv"))
        # self.S.to_csv(os.path.abspath("").replace("\\Code", "\\Prix.csv"))

        Q = self.CopyCar(0)

        Q.Nom = 'Xtime_' + str(p) + 'x_SigmaHebdo_' + Q.Nom
        Q.DerivationName = Q.DerivationName + "_Xtime"
        Q.DerivationLevel = Q.DerivationLevel + 1
        Q.Level1 = "Y"
        Q.Level4 = "Xtime"

        Q.S = pd.DataFrame(nbj, index=df.index)

        return Q
