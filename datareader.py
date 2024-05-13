import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class DataManager:

    def readData(self):
        pass
        data = pd.read_excel("data/alldata.xlsx")
        spalten_zu_addieren = ["Entriegelung Verbr.", "Umstellung Verbr.", "Verriegelung Verbr."]
        spalten_zu_addieren2 = [" Entriegelung Abw.", "Umstellung Abw.", "Verriegelung Abw."]
        spalte_zum_multiplizieren = "Aussentemperatur"

        # Werte addieren
        data["Gesamtabweichung"] = data[spalten_zu_addieren2].sum(axis=1)
        data["Gesamtverbrauch"] = data[spalten_zu_addieren].sum(axis=1)

        # Summe mit Spalte multiplizieren
        data["Verbrauch*Temp"] = data["Gesamtverbrauch"] * data[spalte_zum_multiplizieren]

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
    
    def generateDataSmote(self):
        data = pd.read_excel("data/data1.xlsx")
        """ c1 = data["Datum / Zeit"]
        smote = SMOTE(random_state=42)
        c1_resampled = smote.fit_resample(c1.reshape(-1, 1), c1.reshape(-1, 1))
        data["c1_resampled"] = c1_resampled.reshape(1, -1) """

        X = data.drop(columns=['Datum / Zeit'])  # Annahme: 'Label' ist die Spalte mit den Labels
        y = data['Datum / Zeit']

        # Initialisiere den SMOTE-Algorithmus
        smote = SMOTE(random_state=42)

        # Wende SMOTE auf deine Daten an
        X_smote, y_smote = smote.fit_resample(X, y)

        # Überprüfe die Form deiner neuen Daten
        print("Shape von X_smote:", X_smote.shape)
        print("Shape von y_smote:", y_smote.shape)

        # Erstelle einen DataFrame für die neuen Daten
        new_data = pd.DataFrame(X_smote, columns=X.columns)
        new_data['Datum / Zeit'] = y_smote

        # Füge die neuen Daten den bestehenden Daten hinzu
        combined_data = pd.concat([data, new_data], ignore_index=True)
        print(combined_data.shape)

        # Setze die Optionen für die Anzeige von Pandas DataFrame
        #pd.set_option('display.max_rows', None)  # Anzahl der maximal anzuzeigenden Zeilen auf "None" setzen (zeigt alle Zeilen)
        #pd.set_option('display.max_columns', None)  # Anzahl der maximal anzuzeigenden Spalten auf "None" setzen (zeigt alle Spalten)

        print(combined_data)
        return combined_data

    def analyzeCluster(self, labels, centroids, data ):
        # Daten mit Zuordnung der Clusterlabels erweitern
        clustered_data = pd.DataFrame(data, columns=['X1', 'X2'])
        clustered_data['Cluster'] = labels

        # Charakteristika der Cluster untersuchen
        cluster_characteristics = clustered_data.groupby('Cluster').mean()
        print(cluster_characteristics)

        # Visualisierung der Cluster
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, label='Centroids')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Clustering der Daten mit KMeans')
        plt.legend()
        plt.show()

    def clusterData(self, df):
        #df = pd.read_excel("data/data1.xlsx")  
        print(df) 
        drop = ['Datum / Zeit','Umstellzeit S','Entriegelung S', 'Umstellung S', 'Verriegelung S']
        data = df.drop(drop,axis=1)
        #data = df[['Aussentemperatur', 'Umstellung Verbr.', 'Umstellung Abw.']].copy()
        print(data)
        inertias = []

        for i in range(1,11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
        plt.plot(range(1,11), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

        kmeans = KMeans(n_clusters=5)
        kmeans.fit(data)
        y_kmeans = kmeans.predict(data)
        cluster_zuordnungen = kmeans.labels_
        plt.scatter(data['Aussentemperatur'], data['Umstellung Abw.'], c=cluster_zuordnungen)
        plt.show()

        """ centriods = kmeans.cluster_centers_
        print(data)
        plt.scatter(
            x=data.iloc[:, 0],
            y=data.iloc[:, 1],
            c=inertias, 
            s=50, 
            cmap='Blues',
            alpha = 0.5
        )

        plt.scatter(
            centriods[:, 0], 
            centriods[:, 1], 
            c='black',
            s=200,
            alpha=0.55
        ) """
        
        

        
        
        
        

    
    
    