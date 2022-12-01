import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle

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

st.write(''' ## Comparaison 
---''')

def getAccLlog(model, test_x, test_y):
    prediction = model.predict(test_x)
    prediction = prediction > 0.5
    ll = log_loss(test_y, prediction)
    acc = accuracy_score(test_y, prediction)
    return ll, acc

models=[]

import joblib

for i in models_names:
    filename = f'model-{i}.joblib'
    with open(filename, 'rb') as fo:
        models += [joblib.load(fo)]

for m in models:
    m.fit(train_x, train_y)

conf_mats = []
for m in models:
    conf_mats += [getAccLlog(m, test_x, test_y)]

st.write(conf_mats)

results = pd.DataFrame()
results["model"] = models_names
results = results.join(pd.DataFrame( np.array(conf_mats).reshape((6,2)).tolist(),
                        columns=["log loss", "accuracy"]))
results["time"] = [156, 1, 2, 5, 125, 100]


results.sort_values(["log loss","accuracy"], inplace=True)
results["accuracy"] *= 100
st.write(results)

st.write(''' 
    ## Comparaison plots
    ---

    First some plots about the accuracy and log loss (+ relative accuracy)
''')

# dresults = results.copy()
# dresults[["log loss", "accuracy"]] = results[["log loss", "accuracy"]].diff(axis=0)

results[["mean accuracy diff", "mean log loss diff"]] = results[["accuracy","log loss"]]-results[["accuracy","log loss"]].mean()
fig7 = px.bar(results, x="model", y="mean accuracy diff", color="time",
                barmode='group',
                category_orders={"model":models_names})
fig7.update_layout(barmode='relative')
st.plotly_chart(fig7)

fig8 = px.bar(results, x="model", y="mean log loss diff", color="time",
                barmode='group',
                category_orders={"model":models_names})
fig8.update_layout(barmode='relative')
st.plotly_chart(fig8)

st.write('''
We can deduce that the Machine Learning Algorithm is just as good as the BayesRidge model. They have casi exact accuracy and log loss. This means that they cluster about the same way points. It would be interesting to look under the hood and maybe dive deeper why they behave such.
''')

st.write(''' Now it is your turn to try to write a suspicious email.
We will see how our differents models respond to it and if it is so suspicious !
''' )

import re
from itertools import islice    

txt = st.text_area('Text to analyze', '''Hello, I am hear to ear about what you said last time. I wen to the address but nobody was there ! I can't believed i got scammed. Anyway, i hope this doesn't bother you :) Have fun with your money $ !
I AM JUST PISSED OFF ANY WAY
''')

res = re.split(',| |_|-|!', txt)

# st.write(res)

mots = df.columns.tolist()[:-10]
mots_dict = dict.fromkeys(mots, 0)
for x in res:
    if x in mots:
        mots_dict[x] += 1

ctlt = sum([1 for c in txt if c.isupper()])
ctll = max([sum(1 for c in x if c.isupper()) for x in res])
ctla = np.mean([sum(1 for c in x if c.isupper()) for x in res])

N = len(mots)

# st.write(mots)
# st.write(df.columns.tolist()[-10:-4])

sp = re.split('', txt)

pv_count = sum(c==";" for c in sp)
p_count  = sum(c=="(" for c in sp) + sum(c==")" for c in sp)
b_count  = sum(c=="[" for c in sp) + sum(c=="]" for c in sp)
ex_count = sum(c=="!" for c in sp)
do_count = sum(c=="$" for c in sp) + sum(c=="€" for c in sp)
ha_count = sum(c=="#" for c in sp)

occ = list(mots_dict.values())
occ += [pv_count , p_count, b_count,
        ex_count, do_count, ha_count]

freq = 100*np.array(occ)/sum(occ)
freq = np.concatenate([freq, np.array([ctla, ctll, ctlt])])

freq = freq/train_x.mean()
freq = freq.T
# st.table(freq)

freq = pd.DataFrame(freq).T

st.write(freq)

pred = []

for m in models:
    pred += [m.predict(freq)>0.5]


pred_df = pd.DataFrame(zip(models_names, pred))
pred_df.index = models_names
pred_df.drop(pred_df.columns[0], inplace=True, axis=1)



st.write('Spam or not spam ? :', )

st.write(pred_df)