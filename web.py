import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

#
# with st.sidebar:

#     st.write('''
#         Spam analysis for e-mail De Vinci's security
#         ''')

#     options = st.selectbox(
#         'Browser', ["Dataset", "Preprocessing", "Visualization", "Modeling"]
#         )

st.write(
    """
    # Spam analysis for e-mail De Vinci's security

    Recently Devinci's pole hase been HACKED :{ !

    We have been comissioned to help scan spams in e-mails to help prevent further issues.

    ---
    ---

    ## Data set

    ---

    SPAM E-MAIL DATABASE ATTRIBUTES (in .names format)

    48 continuous real [0,100] attributes of type word_freq_WORD 
    = percentage of words in the e-mail that match WORD,
    i.e. 100 * (number of times the WORD appears in the e-mail) / 
    total number of words in e-mail.  A "word" in this case is any 
    string of alphanumeric characters bounded by non-alphanumeric 
    characters or end-of-string.
    
    6 continuous real [0,100] attributes of type char_freq_CHAR
    = percentage of characters in the e-mail that match CHAR,
    i.e. 100 * (number of CHAR occurences) / total characters in e-mail

    1 continuous real [1,...] attribute of type capital_run_length_average
    = average length of uninterrupted sequences of capital letters

    1 continuous integer [1,...] attribute of type capital_run_length_longest
    = length of longest uninterrupted sequence of capital letters

    1 continuous integer [1,...] attribute of type capital_run_length_total
    = sum of length of uninterrupted sequences of capital letters
    = total number of capital letters in the e-mail

    1 nominal {0,1} class attribute of type spam
    = denotes whether the e-mail was considered spam (1) or not (0), 
    i.e. unsolicited commercial e-mail.  
    """
)

df = pd.read_csv("spambase_prep_norm.csv", sep=";", index_col="Unnamed: 0")
st.dataframe(df)

st.write(
    ''' 
    --- 
    ---

    ## Preprocessing

    ---
    '''
)

st.write(''' 
    ### Missing values
    ---
    With some quick code, we get the only missing value
    being the last row. We simply delete it and move on.
    ''')

st.write(''' 
    ### Encoding
    ---
    Here encoding is needed as all values are as type $object$ instead of $float$ (they are frequencies) but the $spam$ variable.
    ''')

st.code('''
columns = df.columns
for c in columns[:-1]:
    df[c] = df[c].apply(lambda x : float(x))

df["spa"] = df["spa"].apply(lambda x: int(x))
    ''')

st.write(''' 
    ### Outliers and Normalization 
    ---

    Some cleaning is needed :
    ''')

st.code(df.describe())

st.write('''
    ### Outliers
    ---
    Here we get rid of the $1$% outliers in the dataset and set them to the $0.5$% or $99.5th$% depending.
    ''')

st.code(''' 
    columns = df.columns
for c in columns[:-1]:
    df[c] = pd.Series(mstats.winsorize(df[c], limits=[0.01, 0.01])).to_list()
    ''')

st.write(df.describe())

st.write(''' 
    ### Normalization
    ---
    We normalize the data into a new dataframe. It could be useful for interpretation later on.
    ''')
st.write((df-df.min())/(df.max()-df.min()))

###################################### Visualization

st.write('''
    ---
    ---

    ## Data visualization

    ---

    In this data set, we do not have categorical data. Therefore we can simply plot everything as scater plots, violins and so on..
    
    ### Scatter plots


    Something to do in the streamlit app : choose what to plot in 2-3D.
    ''')

fig1 = px.scatter(df, 
                    x = "make",
                    y = "address")

st.plotly_chart(fig1)

st.write(''' 
    We can clearly see some linear behavior even though it is more complex. 

Here are two other figure in respect to $spa$.
''')

fig2 = px.scatter(df, 
                    x = "make",
                    y = "address",
                    color="spa",
                    title="make function of address, colored spa")

st.plotly_chart(fig2)

fig3 = px.scatter(df, 
                    x = "all",
                    y = "our",
                    color="spa",
                    title="all function of our, colored spa")

st.plotly_chart(fig3)

st.write(''' 
    ### Correlation

    ---

    We check for correlation amoung varaibles. It is a bit crowded but the data seems to have some corraltion overall. 
    ''')

corr = df.corr()

fig4 = px.imshow(  corr,
            title="Correlation plot")
fig4.update_layout(
    margin=dict(l=5, r=0, t=30, b=0),
)

st.plotly_chart(fig4)

st.write('''
    We can observe some high positive correlation arround the center. Also, it appears there is no high negative correlation, interesting.

    Let's sum that up and sort it for next plots.
    ''')

corr_sum = corr.sum().sort_values(ascending=False)
corr_cols = corr.columns


st.write(corr_sum[:5])

fig5 = px.histogram(df[corr_cols[1:5]],
                    title="Stacked histogram of the most correlated variables")

st.write('''
    ### Histograms and box plots

    ---

    
    ''')

st.plotly_chart(fig5)

st.write(''' 
    It appears highly correlated variables are mostly centered at $0$. An exeption is $capital\_run\_length\_longest$. One key factor of this is that we took care of the outliers via capping them to the $99$%, this can explain the behavior from the $1+$ side.
    ''')

columns = df.columns

fig6 = px.box(  df[columns[-15:]],
                color="spa",
                title="Box plot of last 14 variables")

st.plotly_chart(fig6)

st.write('''All of a sudden, it is much easier to see that spam have very particular behavior. For exemple, a lot of spam seem to relate to money as of the occurence of the char $\$$, this is feels good as often scam tend to steel money. On the other hand, table does not seem to be a subject of interest.''')


############################# Modeling

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss


import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

import time

st.write(''' ---
---

## Modeling

---

In this part we will do some ML, implement different models to predict if a mail is a spam or not.

We will then process to a comparaison of the different models and conclude.

### Split the data
''')

def getTrainTest(data, p = 0.7, random_state=1, y=""):

    ''' 
    inputs :
    data        : dataframe
    p           : split between train and test sets
    randomstate : seed, only used for consistent results

    outputs :
    train_ : training set of proportion p
    test_  : testing set of proportion 1-p
    '''

    #names the target variable "y"
    data.rename(columns={f"{y}":"y"}, inplace=True)

    #split the data set in test and train
    train_ = data.sample(frac=p, random_state=random_state)
    test_  = data.drop(train_.index) 

    #get y and drops it for x
    train_y = train_.y
    train_x = train_.drop("y", axis=1)

    #get y and drops it for x
    test_y = test_.y
    test_x = test_.drop("y", axis=1)

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = getTrainTest(df, random_state=69, y="spa")

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def getModel(model_type = "linear"):
    ''' 
    inputs :
    model_type : model class  

    outputs : 
    model       : fitted model depending of model_type
    param_grid  : assiocated parameter grid for gridsearch
    '''
    if model_type == "linear":
        model = LinearRegression()
        param_grid = None

    elif model_type == "glm":
        model = PoissonRegressor()
        param_grid = ParameterGrid({
            "alpha":[[x] for x in np.linspace(0, 100, 3)],
            "max_iter":[[x] for x in np.random.randint(1, 1000, (3,))]
        })

    elif model_type == "bayesRidge":
        model = BayesianRidge()
        param_grid = ParameterGrid({
            "n_iter":[[x] for x in np.random.randint(100, 1000, 3)],
            "alpha_1":[[x] for x in np.linspace(1e-9, 1e-3, 3)],
            "alpha_2":[[x] for x in np.linspace(1e-9, 1e-3, 3)],
            "lambda_1":[[x] for x in np.linspace(1e-9, 1e-3, 3)],
            "lambda_2":[[x] for x in np.linspace(1e-9, 1e-3, 3)]
        })

    elif model_type == "randomForest":
        model = RandomForestClassifier()
        param_grid = ParameterGrid({
            "n_estimators":[[x] for x in np.random.randint(50, 150, 3)],
            "criterion":[["gini"],["entropy"],["log_loss"]],
            "max_depth":[[x] for x in np.random.randint(10, 500, (3,))],
            "max_features":[["sqrt"],["log2"]]
        })

    elif model_type == "KNN":
        model = KNeighborsClassifier()
        param_grid = ParameterGrid({
            "n_neighbors":[[x] for x in[1, 2, 3, 5, 10]],
            "weights":[["uniform"],["distance"]],
            "algorithm":[["auto"],["ball_tree"],["kd_tree"]],
            "p":[[x] for x in np.random.randint(1, 53, 3)]
        })

    elif model_type == "MLPCLassifier":
        model = MLPClassifier()
        param_grid = ParameterGrid({
            "hidden_layer_sizes":[(25,),(50,),(100,),(125,),(150,)],
            "activation": [[x] for x in ["identity", "logistic","tanh","relu"]],#note, this is dumb to set all layers to the same activation functions
            "solver":[[x] for x in ["lbfgs","sgd","adam"]]
        })
    
    else : 
        model = sklearn.linear_model.LinearRegression()
        param_grid = None

    return model, param_grid

def getSubBestModel(model, param_grid, X_train, y_train):
    classifier = GridSearchCV(model,
                                    param_grid=param_grid,
                                    # n_iter=10,
                                    cv=5,
                                    verbose=2,
                                    # random_state=420,
                                    n_jobs=-1
                                    )
    classifier.fit(X_train, y_train)
    return classifier.best_estimator_

st.write(''' 
    Here a some models we'll experiment with.
''')

models_names = [
    "MLPCLassifier",
    "linear",
    "glm",
    "bayesRidge",
    "randomForest",
    "KNN"
]

st.write(models_names)

models = []
t = []

prog = 0
for mn in models_names:
    st.progress(prog)
    if mn != "linear":
        start = time.time()
        model, param_grid = getModel(mn)
        models += [getSubBestModel(model, param_grid=param_grid, X_train=train_x, y_train=train_y)]
        end = time.time()
        t += [end-start]
        prog += t/600
prog = 100

lin, _ = getModel()
lin.fit(train_x, train_y)
models += [lin]

st.write(''' ## Comparaison 
---''')

def getAccLlog(model, test_x, test_y):
    prediction = model.predict(test_x)
    prediction = prediction > 0.5
    ll = log_loss(test_y, prediction)
    acc = accuracy_score(test_y, prediction)
    return ll, acc

conf_mats = []
for m in models:
    conf_mats += [getAccLlog(m, test_x, test_y)]

st.write(conf_mats)

results = pd.DataFrame()
results["model"] = models_names
results = results.join(pd.DataFrame( np.array(conf_mats).reshape((6,2)).tolist(),
                        columns=["log loss", "accuracy"]))
results["time"] = t[:1] + [0] + t[1:]


results.sort_values(["log loss","accuracy"], inplace=True)

st.write(results)

st.write(''' 
    ## Comparaison plots
    ---

    First some plots about the accuracy and log loss (+ relative accuracy)
''')

sns.set_color_codes("muted")
sns.barplot(data=results, x="accuracy", y="model", color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(data=results, x='log loss', y='model', color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()

dresults = -(results[["log loss", "accuracy"]].mean() - results[["log loss", "accuracy"]])
dresults["model"] = results["model"]

sns.set_color_codes("muted")
sns.barplot(data=dresults, x="accuracy", y="model", color="b")

plt.xlabel('Accuracy % - mean')
plt.title('Classifier Relative Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(data=dresults, x='log loss', y='model', color="g")

plt.xlabel('Log Loss - mean')
plt.title('Classifier Relative Log Loss')
plt.show()

st.write('''
We can deduce that the Machine Learning Algorithm is just as good as the BayesRidge model. They have casi exact accuracy and log loss. This means that they cluster about the same way points. It would be interesting to look under the hood and maybe dive deeper why they behave such.
''')

st.write(''' Now it is your turn to try to write a suspicious email.
We will see how our differents models respond to it and if it is so suspicious !
''' )