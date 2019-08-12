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

