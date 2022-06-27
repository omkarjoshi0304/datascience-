#iris data set using logistic regression
'''import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv("C:/Users/admin/Downloads/IRIS.csv")

lr=LogisticRegression()

x= df.drop("species",axis=1)
y= df["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

print(accuracy_score(y_test,y_pred))'''


#Iris data set for all algorithms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nb=MultinomialNB()
nn=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)

df=pd.read_csv("C:/Users/admin/Downloads/IRIS.csv")

x= df.drop("species",axis=1)
y= df["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)

lr.fit(x_train,y_train)
l_pred=lr.predict(x_test)

rf.fit(x_train,y_train)
r_pred=rf.predict(x_test)

gbm.fit(x_train,y_train)
g_pred=gbm.predict(x_test)

dt.fit(x_train,y_train)
d_pred=dt.predict(x_test)

sv.fit(x_train,y_train)
s_pred=sv.predict(x_test)

nb.fit(x_train,y_train)
nb_pred=nb.predict(x_test)

nn.fit(x_train,y_train)
n_pred=nn.predict(x_test)



print("logistic regression:",accuracy_score(y_test,l_pred))
print("random forest:",accuracy_score(y_test,r_pred))
print("gradient boosting:",accuracy_score(y_test,g_pred))
print("decision tree ",accuracy_score(y_test,d_pred))
print("naive bayes ",accuracy_score(y_test,nb_pred))
print("neural",accuracy_score(y_test,n_pred))
print("svm",accuracy_score(y_test,s_pred))

'''logistic regression: 0.9777777777777777
random forest: 0.9777777777777777
gradient boosting: 0.9777777777777777
decision tree  0.9777777777777777
naive bayes  0.6
neural 0.24444444444444444
svm 0.9777777777777777

Process finished with exit code 0'''
