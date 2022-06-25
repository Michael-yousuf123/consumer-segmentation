import os
import numpy as np
import pandas as pd
###########################################################
# importing the dataframe into the python environment
###########################################################
def dataframe(path: str = "."):
    """Function to import data and save it
    in to the interim data folder 
    ========================================
    ARGUMENTS:
    path: str ==> the path to dataframe
    ======================================= 
    RETURN:
        data: the output interim df into interim folder
    """
    if os.path.exists(path):
        return pd.read_csv(path)
########################################################
#############################################################