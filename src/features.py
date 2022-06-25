################################################
# Scripts to turn raw data into features for exploration and cluster analysis
################################################
import os
import numpy as np 
import pandas as pd 
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler

def checkDF(df):
    """function for checking of any missing, dupliction
    and abnormal distribution of dataset and return the
    the total number of missing and duplication 
    ======================================================
    ARGUMENTS: the input raw dataframe 
    ======================================================
    RETURN: Total number of missing, Total number of duplication
            and wether there is an abnormality or normality
    """
    cols_missing_no = []
    for col in df.columns.tolist():
        if df[col].isna().sum() != 0:
            print(f"The total missing values of our dataframe is: n\ {df[col].isna().sum()}")
            cols_missing_no.append(col)
        else:
            print("The dataframe column have no any missing values")
        if df.duplicated().sum() !=0:
            print(f"The total missing values of our dataframe is: n\ {df.duplicated().sum()}")
        else:
            print("There are no duplication values observed")
def clean_data(df):
    """this function removes the null and duplicated data points 
    if there is any checked by our preceded function
    =============================================================
    ARGUMENTS:
    =============================================================
    RETURN: cleaned and preprocessed data called interim_data
    """
    dups = df.index.duplicated()
    if dups.sum() > 0:
        df = df[~df.index.duplicated()]
    if df.val.isnull().values.any():
        df.val.fillna(0, inplace=True)

def labelsEncoder(df):
    """the function to convert the categorical and object type 
    dataframe features in to numerical type 
    ===========================================================
    Arguments: 
            x: pandas dataframe
    Return:
            label encoded dataframe """
    from sklearn.preprocessing import LabelEncoder
    features = list(df.select_dtypes(include=['category','object']))
    le = LabelEncoder()
    for cols in features:
        try:
            df[cols] = le.fit_transform(df[cols])
        except:
            print('Error in Encoding!')
    return df

def data_split(df, test_size = 0.2, train_size = 0.8, scaled = False):
    """The function to split our dataframe into train
    and test split
    =================================================
    PARAMETERS:
            df: input dataframe
    =================================================
    RETURN: 
        train.csv: trainset of our input dataframe
        test.csv: testset of our dataframe """
    from sklearn.model_selection import train_test_split
    X = df.drop([''], axis = 1)
    y = df['']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size= test_size, random_state = 42, shuffle = False)
    if scaled == False:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        df_train = pd.concat([X_train, y_train], axis=1)
    
    true_label = ""
    target_label = ""
    pass

    
    

"""if os.path.exists(self.filename) == False:
            headers = list(ordered_dict.keys())
            prev_rec = None
        else:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                prev_rec = [row for row in reader]
            headers = self.merge_headers(headers, list(ordered_dict.keys()))"""
            
""" # inputDataPath = os.path.join(
    #    os.path.dirname(os.path.realpath(__file__)), )
    inputDataPath = '../ProcessedTrafficData'
    outputDataPath = '../NewCleanedData'
    if (not os.path.exists(outputDataPath)):
        os.mkdir(outputDataPath"""
"""  files = os.listdir(inputDataPath)
    for file in files:
        if file.startswith('.'):
            continue
        if os.path.isdir(file):
            continue
        outFile = os.path.join(outputDataPath, file)
        inputFile = os.path.join(inputDataPath, file)
        cleanData(inputFile, outFile)"""
"""# Remove duplicates
        dups = self.data.index.duplicated()
        if dups.sum() > 0:
            warnings.warn(
                'Duplicate values exist, keeping the first occurrence')
            self.data = self.data[~self.data.index.duplicated()] """
"""if self.data.val.isnull().values.any():
            warnings.warn('NaN value(s) present, coercing to zero(es)')
            self.data.val.fillna(0, inplace=True)"""
            
"""def normalize_data(data_matrix):
        """
        # Normalize data to have mean 0 and variance 1 for each column

        # :param data_matrix: matrix of all data
        # :return: normalized data
        # """
        # try:
        #     mat_size = np.shape(data_matrix)
        #     for i in range(0, mat_size[1]):
        #         the_column = data_matrix[:, i]
        #         column_mean = sum(the_column)/mat_size[0]
        #         minus_column = np.mat(the_column-column_mean)
        #         std = np.sqrt(np.transpose(minus_column)*minus_column/mat_size[0])
        #         data_matrix[:, i] = (the_column-column_mean)/std
        #     return data_matrix
        # except Exception as e:
        #     print(e)
        # finally:
        #     pass"""

"""def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler"""
        
"""def normalize_data_costs(self, data_processed, normalization, analysis_fields):
        if normalization == "gross floor area":
            data = pd.read_csv(self.locator.get_total_demand())
            normalization_factor = sum(data['GFA_m2'])
            data_processed = data_processed.apply(
                lambda x: x / normalization_factor if x.name in analysis_fields else x)
        elif normalization == "net floor area":
            data = pd.read_csv(self.locator.get_total_demand())
            normalization_factor = sum(data['Aocc_m2'])
            data_processed = data_processed.apply(
                lambda x: x / normalization_factor if x.name in analysis_fields else x)
        elif normalization == "air conditioned floor area":
            data = pd.read_csv(self.locator.get_total_demand())
            normalization_factor = sum(data['Af_m2'])
            data_processed = data_processed.apply(
                lambda x: x / normalization_factor if x.name in analysis_fields else x)
        elif normalization == "building occupancy":
            data = pd.read_csv(self.locator.get_total_demand())
            normalization_factor = sum(data['people0'])
            data_processed = data_processed.apply(
                lambda x: x / normalization_factor if x.name in analysis_fields else x)
        return data_processed """