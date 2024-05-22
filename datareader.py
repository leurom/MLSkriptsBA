import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier

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
        drop = ['Datum / Zeit']
        data = data.drop(drop, axis=1)
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
    
    def exploreData(self, data):
        # Laden des Iris-Datensatzes von Seaborn
        #data = sns.load_dataset('iris')

        # Anzeigen der ersten paar Zeilen des Datensatzes
        print("Die ersten fünf Zeilen des Datensatzes:")
        print(data.head())
        print(data.info())
        # Anzeigen der statistischen Zusammenfassung der Daten
        print("\nStatistische Zusammenfassung des Datensatzes:")
        print(data.describe())

        # Visualisierung der Verteilung der Klassen
        sns.countplot(x='Aussentemperatur', data=data)
        plt.title('Verteilung der Klassen')
        plt.show()

        # Visualisierung der Verteilung der Merkmale
        data.drop('Aussentemperatur', axis=1).hist(edgecolor='black', linewidth=1.2, figsize=(12, 8))
        plt.suptitle("Verteilung der Merkmale")
        plt.show()

        # Paarplot für Merkmale mit Klassenfarben
        """ sns.pairplot(data, hue='Aussentemperatur')
        plt.title("Pairplot der Merkmale mit Klassenfarben")
        plt.show() """

        # Berechnung der Korrelationen
        correlation_matrix = data.corr()

        # Visualisierung der Korrelationen als Heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Korrelationsmatrix')
        plt.show()

    
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
        clustered_data = pd.DataFrame(data, columns=['Aussentemperatur', 'Gesamtverbrauch'])
        clustered_data['Cluster'] = labels
        data = data[['Aussentemperatur', 'Gesamtverbrauch']]

        # Charakteristika der Cluster untersuchen
        cluster_characteristics = clustered_data.groupby('Cluster').mean()
        print(cluster_characteristics)
        # Visualisierung der Cluster
        plt.figure(figsize=(8, 6))
        plt.scatter(data['Aussentemperatur'], data['Gesamtverbrauch'], c=labels, cmap='viridis', s=50, alpha=0.5)
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='*', s=200, label='Centroids')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Clustering der Daten mit KMeans')
        plt.legend()
        plt.show()

    def clusterData(self, df):
        #df = pd.read_excel("data/data1.xlsx")  
        print(df) 
        scaler = StandardScaler()
        #data = scaler.fit_transform(df)
        data = df[['Aussentemperatur', 'Gesamtverbrauch']]
        drop = ['Datum / Zeit','Umstellzeit S','Entriegelung S', 'Umstellung S', 'Verriegelung S']
        #data = df.drop(drop,axis=1)
        #data = df[['Aussentemperatur', 'Umstellung Verbr.', 'Umstellung Abw.']].copy()
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
        print(centroids)
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        y_kmeans = kmeans.predict(data)
        cluster_zuordnungen = kmeans.labels_
        plt.scatter(data['Aussentemperatur'], data['Gesamtverbrauch'], c=cluster_zuordnungen)
        plt.show()
        return labels, centroids
    

    def logRegression(self, data):
        y = data['Umstellung S']
        x = data.drop(['Umstellung S'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
        logreg = LogisticRegression(random_state=16)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)

        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        cnf_matrix

        target_names = ['1', '0']
        print('Classification Report:')
        print(classification_report(y_test, y_pred, target_names=target_names))

    def neuralNetwork(self, data):
        y = data['Umstellung S']
        x = data.drop(['Umstellung S'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=200, random_state=42)

        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        target_names = ['1', '0']
        print('Classification Report:')
        print(classification_report(y_test, y_pred, target_names=target_names))


        
        
        