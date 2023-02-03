import pickle
import json
import numpy as np

class cgpa_model():

    def __init__(self,GRE,TOEFL,Rating,SOP,LOR,Admit):
        self.GRE = GRE
        self.TOEFL = TOEFL
        self.Rating = Rating
        self.SOP = SOP
        self.LOR = LOR
        self.Admit = Admit

    def load_model(self):
        self.l_model =pickle.load(open("cgpa.pkl",'rb'))
        self.l_data =json.load(open("data_cgpa.json",'r'))


    def get_predicted_cgpa(self):
        self.load_model()

        test_array = np.zeros(len(self.l_data['columns']))
        test_array[0] = self.GRE
        test_array[1] = self.TOEFL
        test_array[2] = self.Rating
        test_array[3] = self.SOP
        test_array[4] = self.LOR
        test_array[5] = self.Admit


        print('Test Array :', test_array)
        #model = self.l_model
        predicted_cgpa = np.around(self.l_model.predict([test_array])[0],2)
        print('The predicted Charges is RS.:',predicted_cgpa)
        print()
        return predicted_cgpa
