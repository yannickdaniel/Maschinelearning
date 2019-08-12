import pandas

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


#url = "https://raw.githubsercontent.com/jbrownlee/Datasets/master/iris.csv"
url = "/home/yannick/Schreibtisch/machinelearning/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#shape of the dataset: (150, 5)
shape = dataset.shape
print(shape)

#Peek at the Data:
head_num = 20
head = dataset.head(head_num)
print(head)

#basic statistics (count, mean, std, min, 25%, 50%, 75%, max)
describe = dataset.describe()
print(describe)

#Class Distribution (Gruppieren)
groups = dataset.groupby('class').size()
print(groups)

#Plots
dataset.plot(kind= 'box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Histogram
dataset.hist()
plt.show()

#Scatterplots
scatter_matrix(dataset)
plt.show()

#Evaluate some Algorithms
#Aufteilung in Trainings- und Testdaten
scoring = 'accuracy'
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = validation_size, random_state = seed)


#Build Models:
#Welches Model ist am besten geeignet?
#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes (NB).
#Support Vector Machines (SVM).

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class= 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#-> Es werden Werte für die Genauigkeit des jeweiligen Algorithmus errechnet
#-> Support Vector Machines scheint hier die beste Lösung zu sein

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklables(names)
plt.show()
