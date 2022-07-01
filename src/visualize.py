##############################################################################
# Producing the Sripts to create exploratory and results oriented visualizations
###############################################################################
import os
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

def pairplot(data, figures: str = '.'):
    """Function to create the scatter plot for the comparison of 
    numerical variables with that of non numerical categorical variables
    and we use the nominal variable as hue to so as to depict the difference
    between countries in indicators 

    Args:
        data (pandas dataframe): _description_
    """
    if not os.path.exists(figures):
        os.makedirs(figures)
    else:
        print("Directory already exists")
    fig = px.scatter_matrix(data, dimensions=["water_prc", "elect_prc", "tel_prc"], color="Districts")
    # fig.write_image("figures/pair.png")
    fig.show()

def component_no(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    # summarizing of data with multiple features are with covarience and correlation coefficient
    # for single variable the summaries are with mean, mode and standard deviations
    input = data.drop(['Districts'], axis = 1)
    pca = PCA().fit(input)
    plt.figure(figsize = (8, 4))
    components = np.arange(1, 4, 1)
    variance = np.cumsum(pca.explained_variance_ratio_)
    ylim = (0.0, 1.1)
    plt.plot(components, variance, marker = 'o', linestyle = '--', color = 'r')
    plt.xticks(np.arange(1, 4, step=1))
    plt.xlabel(('Number of Components'))
    plt.title("Number of Components to explain Variance")
    plt.ylabel(str("% Cumulative Variance"))
    plt.axhline(y = 0.95, color = 'b', linestyle = '-')
    plt.text(2, 0.94, '95% Variance cut-off', color = 'r', fontsize = 10)
    return plt.show()

def kplots(df, n_init= 10, plot=True):
    """"""
    wcss = []
    for i in range(1, 11):
        model = KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=n_init, random_state=42)
        model.fit(df)
        wcss.append(model.inertia_)
    k = [i*100 for i in np.diff(wcss,2)].index(min([i*100 for i in np.diff(wcss,2)]))
    if plot:
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=wcss,
                            mode='lines',
                            name='lines'))
        fig.update_layout(title='Elbow Method for k',
                   xaxis_title='Number of Clusters',
                   yaxis_title='Distortions')
        fig.add_vline(x=k, line_dash="dot",
              annotation_text="k = "+str(k), 
              annotation_position="top right")
        fig.show()
    return k 