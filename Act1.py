import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Titanic = pd.read_csv("Titanic_Dataset.csv")

Titanic.head()
Titanic.columns
Titanic.shape
Titanic.isnull().sum()
plt.figure(figsize=(10, 6))

sns.heatmap(Titanic.isnull(), cmap="spring")

plt.show()
Titanic.drop("Cabin", axis=1,inplace=True)
Titanic.head()
Titanic.dropna(inplace=True)

Titanic.isnull().sum()
plt.figure(figsize=(10, 6))

sns.heatmap(Titanic.isnull(), cbar=False)

plt.show()
pd.get_dummies(Titanic["Sex"]).head()

sex = pd.get_dummies(Titanic['sex'],drop_irst=True)
sex.head(4)

pd.get_dummies(Titanic["Emabarked"]).head()
embarked = pd.get_dummies(
    Titanic["Emabarked"],
    drop_first=True
)

embarked.head(4)
pclass = pd.get_dummies(
    Titanic["Pclass"],
    drop_first=True
)

pclass.head(4)

Titanic = pd.concat(
    [Titanic, sex, embarked,pclass],
    axis=1
)
Titanic.head()