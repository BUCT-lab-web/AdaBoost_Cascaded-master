from sklearn.ensemble import RandomForestClassifier
from extra import rotation_forest
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from Rotation import rotation_adaboost 

def all_model(fx_train,fx_test,y_train,y_test):

    result=[]
    classifier = rotation_forest.RotationForestClassifier(n_features_per_subset=10)
    classifier.fit(fx_train, y_train)
    y_pred2 = classifier.predict(fx_test)
    pre2=np.mean(np.equal(y_pred2,y_test))
    result.append(pre2)
    #adaboost rf
    adaherf = rotation_adaboost.RotationAdaBoostClassifier()
    adaherf = adaherf.fit(fx_train, y_train)
    y_pred6 = adaherf.predict(fx_test)
    pre8=np.mean(np.equal(y_pred6,y_test))
    result.append(pre8)
    classifier = RandomForestClassifier(n_estimators = 40 )
    classifier.fit(fx_train, y_train)
    y_pred1 = classifier.predict(fx_test)
    pre1=np.mean(np.equal(y_pred1,y_test))
    # print('Rondom_Pre:',pre1)
    result.append(pre1)
    # #svm
    # classifier=svm.SVC(kernel='rbf')
    # classifier.fit(fx_train, y_train)
    # y_pred4 = classifier.predict(fx_test)
    # pre4=np.mean(np.equal(y_pred4,y_test))
    # # print('SVM:',pre4)
    # result.append(pre4)

    #Bagging
    classifier=BaggingClassifier(RandomForestClassifier(),n_estimators=10)
    classifier.fit(fx_train, y_train)
    y_pred5 = classifier.predict(fx_test)
    pre5=np.mean(np.equal(y_pred5,y_test))
    # print('bagging:',pre5)
    result.append(pre5)

    #Bagging
    classifier=BaggingClassifier(rotation_forest.RotationForestClassifier(n_features_per_subset=10),n_estimators=10)
    classifier.fit(fx_train, y_train)
    y_pred5 = classifier.predict(fx_test)
    pre5=np.mean(np.equal(y_pred5,y_test))
    # print('bagging:',pre5)
    result.append(pre5)

    classifier=AdaBoostClassifier(RandomForestClassifier(),n_estimators=10)
    classifier.fit(fx_train, y_train)
    y_pred5 = classifier.predict(fx_test)
    pre5=np.mean(np.equal(y_pred5,y_test))
    # print('bagging:',pre5)
    result.append(pre5)
    

    return result






