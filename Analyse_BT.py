from pandas import read_csv, to_datetime, DataFrame
import glob
import os
import re
import pandas as pd
from functools import reduce
import pathlib
import numpy as np

class BackTest:
    def __init__(self):

        self.Valo = DataFrame
        self.Prediction = DataFrame
        self.Weight = DataFrame

        self.Y_name = ''
        self.Parametre_name = ''
