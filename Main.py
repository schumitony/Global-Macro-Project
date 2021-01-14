from DataM import DataM
from BackTest import BackTest
import re
import os
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
            if False:
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
        list_model = ['XGBRegressor', 'RandForestReg']
        fut = ["Bund1", "EURUSD_Fut1", "TNote_10_Fut1", "TBond_30_Fut1"]
        deriv = ["Raw_Return", "Raw_Normalized_Return"]
        hor = [5, 10, 22]

        list_bt, h = Data.listBT(fut=fut, deriv=deriv, horizon=hor)
        # list_bt, h = Data.listBT(fut=fut, horizon=hor, deriv=deriv)
        # list_bt, h = Data.listBT("Bund1")

        Data.All_bt(list_bt, h, 'blend_test_cv', 'weight_ls_param_rescaling', 'score_ada', list_model, 0.25)
        Data.All_bt(list_bt, h, 'time_cv', 'weight_ls_param_rescaling', 'score_ada', list_model, 0.25)
        Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls_param_rescaling', 'score_ada', list_model, 0.25)
        
        Data.All_bt(list_bt, h, 'blend_test_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', list_model, 0.25)
        Data.All_bt(list_bt, h, 'time_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', list_model, 0.25)
        Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls_param_rescaling', 'neg_mean_squared_error', list_model, 0.25)

        Data.All_bt(list_bt, h, 'blend_test_cv', 'weight_ls', 'score_ada', list_model, 0.25)
        Data.All_bt(list_bt, h, 'time_cv', 'weight_ls', 'score_ada', list_model, 0.25)
        Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls', 'score_ada', list_model, 0.25)

        Data.All_bt(list_bt, h, 'blend_test_cv', 'weight_ls', 'neg_mean_squared_error', list_model, 0.25)
        Data.All_bt(list_bt, h, 'time_cv', 'weight_ls', 'neg_mean_squared_error', list_model, 0.25)
        Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls', 'neg_mean_squared_error', list_model, 0.25)

        # Poids Max 100 pct
        # Data.All_bt(list_bt, h, 'time_cv', 'weight_ls', 'score_ada', 1)
        # Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls', 'score_ada', 1)

        # Data.All_bt(list_bt, h, 'time_cv', 'weight_ls', 'neg_mean_squared_error', 1)
        # Data.All_bt(list_bt, h, 'blend_cv', 'weight_ls', 'neg_mean_squared_error', 1)
