##############################################################################
# Producing the Sripts to create exploratory and results oriented visualizations
###############################################################################
import os
import pandas as pd 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')
import seaborn as sns

def visual_plot(df, )
def scatterplot(df, x: str, y: str, outPutPath: str = "."): 
    """ Function to create the scatter plot for our dataframe including the
    directory to save the images from the outputs of scatter function  
    =============================================================
    ARGUMENTS
    =============================================================
    df ([type]): our input dataframe 
    x (str): the featrure in the x axis
    y (str): the feature in the y axis
    outPutPath (str): our output directory
    =============================================================
    RETURNS: the saved scatterplot images in our working environment
    =============================================================
    """
    if not os.path.exists(outPutPath):
        os.makedirs(outPutPath)
    else:
        print("Directory already exists")
    Scatter = ("C:/Users/miki/Desktop/VSML/cluster-machine-learning/reports/figures/ScatterPlots")
    outPuts = os.path.join(outPutPath, Scatter)
    fig, ax = plt.subplots(figsize= (14, 9))
    graph = sns.scatterplot(ax=ax, x=x, y=y, data=df, s=325, alpha=0.5, hue="country")
    box = ax.get_position()
    plt.legend(markerscale=2)
    img = os.path.join(outPuts, "{}v{}.png".format(x, y))
    plt.savefig(img)
    plt.show()
    return img
def histogram(df, x, outPutPath: str = "."):
    """ 
    Function to create the summary statistics histogram plots for 
    our input dataframe and save the results into the figure output
    paths
    Args:
        df(dataframe): input data
        x (int or float): Dataframe features
        outPutPath (str): _description_
    return:
        Histogram plot
    """
    
    HistoGramPlot = ("C:/Users/miki/Desktop/VSML/cluster-machine-learning/reports/figures/HistoGram")
    if not os.path.exists(outPutPath):
        os.makedirs(outPutPath)
    else:
        print("Directory already exists")
    result = os.path.join(outPutPath, HistoGramPlot)
    if x.dtype == 'int64' or x.dtype == 'float64':
        output = sns.displot(data=df, x=x, bins = 10, kind='hist',hue_norm=None, kde = True, color=None, col_wrap=None,height=5, aspect=1, facet_kws=None)
        y = plt.ylabel('Frequency', size = 17)
        plt.grid = False
        plt.xticks(size = 17)
        plt.yticks(size = 17)
        plt.title(x.name, size = 19) 
        img = os.path.join(result, "{}v{}.png".format(x, y))
        plt.savefig('C:/Users/miki/Desktop/VSML/cluster-machine-learning/reports/figures/HistoGramPlot/img.png')
        return img
def pairplot(data, OutPutPath: str = '.'):
    """Function to create the scatter plot for the comparison of 
    numerical variables with that of non numerical categorical variables
    and we use the nominal variable as hue to so as to depict the difference
    between countries in indicators 

    Args:
        data (pandas dataframe): _description_
    """
    if not os.path.exists(OutPutPath):
        os.makedirs(OutPutPath)
    else:
        print("Directory already exists")
    PairPlot = ("C:/Users/miki/Desktop/VSML/cluster-machine-learning/reports/figures/PairPlots")
    outputs = os.path.join(OutPutPath, PairPlot)
    graph = sns.pairplot(data = data,  x_vars=["IS", "IU_Per100", "MCS_Per100", "TEG"],
                        y_vars=["IS", "IU_Per100", "MCS_Per100", "TEG"], hue="country", kind = 'scatter')
    fig = os.path.join(outputs, "{}.png".format(graph))
    graph.figure.savefig("C:/Users/miki/Desktop/VSML/cluster-machine-learning/reports/figures/PairPlots/pairplot.png")
    return fig
def correlationPlot():
    pass

