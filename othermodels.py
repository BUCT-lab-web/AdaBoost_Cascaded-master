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


    adaherf = rotation_adaboost.RotationAdaBoostClassifier()
    adaherf = adaherf.fit(fx_train, y_train)
    y_pred6 = adaherf.predict(fx_test)
    pre8=np.mean(np.equal(y_pred6,y_test))
    result.append(pre8)
    
    
    return result






