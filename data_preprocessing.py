import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('Copy of sonar data.csv', header=None)

sonar_data.head()

# number of rows and columns
sonar_data.shape

sonar_data.columns

sonar_data.info()

sonar_data.describe() 

sonar_data.describe(include= 'object')

sonar_data.isnull().sum()

sonar_data[60].value_counts()

sonar_data.groupby(60).mean()

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)

print(Y)

sns.countplot(x = Y)

sns.heatmap(sonar_data.isnull())


