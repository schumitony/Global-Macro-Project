from functools import reduce
import pandas as pd

class Utilitaire:

    @staticmethod
    def dict_list_to_df(data):

        if isinstance(data, dict):
            ll = list()
            for k1, v in data.items():
                if isinstance(data[k1], pd.DataFrame):
                    ll.append(v)
                else:
                    ll.append(v.to_frame(name=k1))

        elif isinstance(data, list):
            ll = [k for k in data]

        if 'll' in locals():
            df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='outer'), ll)

        return df





    @staticmethod
    # Change une liste ou Dict de Serie en liste ou Dict de Dataframe
    def to_frame(data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(data[k], pd.Series):
                    data[k] = data[k].to_frame()

        elif isinstance(data, list):
            data = [k.to_frame() for k in data]

        return data
