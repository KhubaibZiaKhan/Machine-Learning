import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Cars = pd.read_csv("Cars.csv")

Cars.head()
Cars.columns
Cars.shape
Cars.isnull().sum()
plt.figure(figsize=(10, 6))

sns.heatmap(Cars.isnull(), cmap="spring")

plt.show()
Cars.drop("Cabin", axis=1,inplace=True)
Cars.head()
Cars.dropna(inplace=True)

Cars.isnull().sum()
plt.figure(figsize=(10, 6))

sns.heatmap(Cars.isnull(), cbar=False)

plt.show()
pd.get_dummies(Cars["Sex"]).head()

sex = pd.get_dummies(Cars['sex'],drop_irst=True)
sex.head(4)

pd.get_dummies(Cars["Emabarked"]).head()
embarked = pd.get_dummies(
    Cars["Emabarked"],
    drop_first=True
)

embarked.head(4)
pclass = pd.get_dummies(
    Cars["Pclass"],
    drop_first=True
)

pclass.head(4)

Cars = pd.concat(
    [Cars, sex, embarked,pclass],
    axis=1
)
Cars.head()