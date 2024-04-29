import pandas as pd

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