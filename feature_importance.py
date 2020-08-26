import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


df = pd.read_csv('task_data.csv')
X = df.iloc[:len(df.values), 2:]
y = df.iloc[:len(df.values), 1]

#split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=21)
model = LogisticRegression()
model2 = svm.SVC(kernel='linear', C=2, gamma=1)

# # training the models
model.fit(X_train, y_train)
model2.fit(X_train, y_train)

# calculate importance of features for logistic regression
importance_log = model.coef_[0]
importance_log = abs(importance_log)

# calculate importance of features for svmn
importance_svm = model2.coef_[0]
importance_svm = abs(importance_svm)


# print feature importance
for i in range(len(importance_log)):
    print('########## Feature %d ###########' % (i + 1))
    print('Logistic_reg: score: %.5f' % importance_log[i])
    print('SVM: feature: score: %.5f' % importance_svm[i])

# plot feature importance
wth= 0.3  # bar width
graph = pyplot.subplot(111)
graph.bar([x - wth / 2 for x in range(len(importance_log))], importance_log, width=wth, align='center', color='red', label ='Logistic reg')
graph.bar([x + wth / 2 for x in range(len(importance_log))], importance_svm, width=wth, align='center', color='blue', label='svm')
pyplot.title('Sensor data importance')
pyplot.xlabel('Sensor number')
pyplot.ylabel('Sensor importance score')
pyplot.legend()
pyplot.show()
