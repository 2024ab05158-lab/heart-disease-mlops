#Load & Clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/heart.csv")
df.head()
df.replace("?", np.nan, inplace=True)
df = df.astype(float)

# Binary target: 1 = disease present, 0 = absent
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

df.info()


plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Class Distribution (Heart Disease Presence)")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()


