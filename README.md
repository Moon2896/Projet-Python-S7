# Projet-Python-S7
---
This projec takes place in the course "Python for data analysis" at ESILV DIA M1.

This project is made of 3 notebooks and 1 python file. The first takes care of what they are called. The latter is an equivalent of the 3 combined put into streamlit.

To make this work, you might need to start the .py using the venv.

This can be done as follow : 
1- get in directory
2- get to .env/Scripts
3- "activate" into cmd prompt
4- got back in directory
5- run $streamlit run web.py
6- enjoy

---
## Preprocessing
---
There is not a lot of preprocessing involved here.
Encoding values into floats.
We set the min and max 0.5% to their closest value that is in the [0.05, 0.995] interval.
We then normalize the data.

---
## Data Visualization
---

It is quite difficult to make the data obvious, visualize it a way that would help understand.

There are a few plots:
Scatter
Correlation
Histogram
Boxplot

---
## Data modeling
---

We first split the data using a 70/30 split.

We train a few models : 
    MLPCLassifier
    linear
    glm
    bayesRidge
    randomForest
    KNN

Using gridsearchcv from sklearn model selection. It takes a few minutes to cross validate the different models, therefore we save them to avoid this later on.

Model comparaison:
We compare each model by accuracy and log-loss. We plot this and give the user the possibility to enter the exemple of an email. 
---