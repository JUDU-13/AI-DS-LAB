from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

data = load_iris() #download csv file for easy coding
df = pd.DataFrame(data=data.data, columns=data.feature_names)

sam1 = df.loc[:, "sepal length (cm)"]
sam2 = df.loc[:, "petal width (cm)"]

cov =(np.cov(sam1, sam2)[0][1])
cor =(np.corrcoef(sam1, sam2)[0][1])
print(f"Sepal Length & Petal Width of Iris Flower\nCovariance: {cov}\nCorrelation:{cor}")
