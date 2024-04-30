import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DataReader:

    def readData(self):
        pass
        data = pd.read_excel("data/data1.xlsx")
        #print(data)
        #print(data.keys())
        """ data["Umstellzeit"] = data["Umstellzeit"].astype(float)
        data["Aussentemperatur"] = data["Aussentemperatur"].astype(float)
        data[" Entriegelung Abw."] = data[" Entriegelung Abw."].astype(float)
        data["Entriegelung Verbr."] = data["Entriegelung Verbr."].astype(float)
        data["Umstellung Abw."] = data["Umstellung Abw."].astype(float)
        data["Umstellung Verbr."] = data["Umstellung Verbr."].astype(float)
        data["Verriegelung Abw."] = data["Verriegelung Abw."].astype(float)
        data["Verriegelung Verbr."] = data["Verriegelung Verbr."].astype(float) """
        return data
    
    def generateData(self,X,y):
        X, y = make_classification(n_classes=2, class_sep=2,
        weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
        n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
        print('Original dataset shape %s' % Counter(y))
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        print('Resampled dataset shape %s' % Counter(y_res))

    def generateDataSmote(self,X,y):
        # Initialisiere und generiere synthetische Daten
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Kombiniere ursprüngliche und synthetische Daten
        X_combined = np.concatenate([X, X_resampled], axis=0)
        y_combined = np.concatenate([y, y_resampled], axis=0)

        print(X_combined, y_combined)

    def clusterData(self):
        df = pd.read_excel("data/data1.xlsx")   
        #y = df.columns[10]
        #x = df.drop(df.columns[10], axis=1)   
        #data = list(zip(x, y))
        x = df[['Aussentemperatur', 'Umstellung Verbr.']].copy()
        inertias = []

        for i in range(1,11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(x)
            inertias.append(kmeans.inertia_)

            plt.plot(range(1,11), inertias, marker='o')
            plt.title('Elbow method')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.show()