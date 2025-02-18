import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r'C:\Users\jeetm\Desktop\ML-Finlatics\wine_data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#1.	What is the most frequently occurring wine quality? What is the highest number in and the lowest number in the quantity column?
quality_counts=data['quality'].value_counts()
most_freq=quality_counts.idxmax()
highest_quality=data['quality'].max()
lowest_quality=data['quality'].min()
print(f"1-> Most frequent wine quality {most_freq} and its Range {lowest_quality,highest_quality}")

#2.How is `fixed acidity` correlated to the quality of the wine? How does the alcohol content affect the quality? How is the `free Sulphur dioxide` content correlated to the quality of the wine?
correlation=data.corr()['quality']
fixed_acidity_corr=correlation['fixed acidity']
alcohol_corr=correlation['alcohol']
free_sulphur_corr=correlation['free sulfur dioxide']
print(f"2-> fixed_acidity,alcohol,free sulphur dioxide correlation with wine quality {fixed_acidity_corr,alcohol_corr,free_sulphur_corr}  ")

#3.	What is the average `residual sugar` for the best quality wine and the lowest quality wine in the dataset?
avg_sugar_best=data[data['quality']==highest_quality]['residual sugar'].mean()
avg_sugar_lowest=data[data['quality']==lowest_quality]['residual sugar'].mean()
print(f"3-> Average residual sugar for best and lowest quality wines are {avg_sugar_best,avg_sugar_lowest}")

#4.Does `volatile acidity` has an effect over the quality of the wine samples in the dataset?
volatile_acidity_corr = correlation['volatile acidity']
print(f"4-> voltile acidity correlation with wine quality is {volatile_acidity_corr}")
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['volatile acidity'], y=data['quality'], hue=data['quality'], palette='viridis')
plt.title('Effect of Volatile Acidity on Wine Quality', fontsize=14)
plt.xlabel('Volatile Acidity (g/L)', fontsize=12)
plt.ylabel('Quality', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

#5.	Train a Decision Tree model and Random Forest Model separately to predict the Quality of the given samples of wine. Compare the Accuracy scores for both models.
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini',random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
print(f"5-> Decision Tree Accuracy {accuracy} and its confusion matrix")
print(cm)

from sklearn.ensemble import RandomForestClassifier
classifier1=RandomForestClassifier(n_estimators=50,criterion='gini',random_state=0)
classifier1.fit(x_train,y_train)
y_pred1=classifier1.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix
rf_cm=confusion_matrix(y_test, y_pred1)
rf_accuracy=accuracy_score(y_test, y_pred1)
print(f"Random Forest Accuracy {rf_accuracy} and its confusion matrix")
print(rf_cm)

