import numpy as np
import pickle

class hpp():
    def __init__(self,data):
        self.data = data

    def load_model(self):
        with open(r'artifacts/model.pkl','rb')as file:
            self.model = pickle.load(file)

    def predict(self):
        self.load_model()

        SL = float(self.data['SL'])
        SW = float(self.data['SW'])
        PL = float(self.data['PL'])
        PW = float(self.data['PW'])
        arr = np.array([[SL,SW,PL,PW]])
        
        print(arr)
        
        result = self.model.predict(arr)
        

        if result == 0:
            print("Iris-Setosa")
        if result == 1:
            print("Iris-Veriscolor")
        if result == 2:
            print("Iris-Verginica")

        return result

# if __name__ == "__main__":
#     data={
#     "SL":7.6,
#     "SW":5.2,
#     "PL":3.2,
#     "PW":2.3
#     }

# species_obj=species(data)

# species_obj.predict()

