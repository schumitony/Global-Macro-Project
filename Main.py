from DataM import DataM
from BackTest import BackTest
import re
import os
import time
from LogFiles import log
import pickle

class Main:

    if __name__ == '__main__':
        # logmsg = log(path="Log", create=True)
        # logmsg.write("eee")
        # logmsg.write("543")

        # logmsg = log(path="JeParle", nom="MaisOui", create=True)
        # algo_name = "ee"
        # logmsg.write('Best ' + algo_name + ': %f using %s' % (1, 5))

        if True:
            if True:
                Data = DataM()
                Data.Loading()
                Data.ClearData(creatCSV=True)

                with open(os.path.abspath("").replace("Program_ML\\Code", "") +'Raw_Data.pkl', 'wb') as output:
                    pickle.dump(Data, output, pickle.HIGHEST_PROTOCOL)
            else:
                Data = pickle.load(open(os.path.abspath("").replace("Program_ML\\Code", "") + 'Raw_Data.pkl', 'rb'))
            Data.Derivation(creatCSV=True)
        else:
            Data = DataM(Load=True)

        # list_model = ['Neuronal']
        # fut = ["Bund1"]
        # deriv = ["Raw_Return"]
        # hor = [5]

        # list_model = ['Neuronal', 'XGBRegressor', 'RandForestReg']
        # list_model = ['XGBRegressor', 'RandForestReg']
        # fut = ["Bund1", "Stoxx50_Fut1"]
        # deriv = ["Raw_Return", "Raw_PositiveReturn"]
        # hor = [1, 7, 30]

        Group_stra = Data.listBT_II()
        for nom, s in Group_stra.items():
            Data.All_bt(nom, s)


        # Data.All_bt(Stra, 'blend_test_cv', 'weight_ls_param_rescaling', 'score_ada', 0.25)
        # Data.All_bt(Stra, 'time_cv', 'weight_ls_param_rescaling', 'score_ada', 0.25)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls_param_rescaling', 'score_ada', 0.25)
        #
        # Data.All_bt(Stra, 'blend_test_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', 0.25)
        # Data.All_bt(Stra, 'time_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', 0.25)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', 0.25)
        #
        # Data.All_bt(Stra, 'blend_test_cv', 'weight_ls', 'score_ada', 0.25)
        # Data.All_bt(Stra, 'time_cv', 'weight_ls', 'score_ada', 0.25)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls', 'score_ada', 0.25)
        #
        # Data.All_bt(Stra, 'blend_test_cv', 'weight_ls', 'neg_mean_squared_error', 0.25)
        # Data.All_bt(Stra, 'time_cv', 'weight_ls', 'neg_mean_squared_error', 0.25)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls', 'neg_mean_squared_error', 0.25)




        # Poids Max 100 pct
        # Data.All_bt(Stra, 'time_cv', 'weight_ls', 'score_ada', 1)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls', 'score_ada', 1)

        # Data.All_bt(Stra, 'time_cv', 'weight_ls', 'neg_mean_squared_error', 1)
        # Data.All_bt(Stra, 'blend_cv', 'weight_ls', 'neg_mean_squared_error', 1)
