from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Loading data
irisData = load_iris()
 
# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

#when test_size is low more training data (its best to stay in between 0.2 and 0.3)

a=int(input("Enter the number for K value:"))
knn = KNeighborsClassifier(n_neighbors=a)
print("Number of datas for Preiction is:",len(X_test))

#training
knn.fit(X_train, y_train)
print("Prediction Accuracy:",knn.score(X_test, y_test)*100)

# Predict on dataset which model has not seen before
x=knn.predict(X_test)
plt.ylabel("PREDICTION")
plt.xlabel("SlNo")
plt.plot(x)
print("Actual:",y_test)
print("Predicted:",knn.predict(X_test))
