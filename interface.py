from flask import Flask,render_template,request,jsonify
#from cgpa_class import cgpa_model
import numpy as np
import pickle
import pandas as pd
#import sklearn


print('Welcome To Prediction App')

with open('cgpa.pkl','rb') as f:
    l_model = pickle.load(f)

app=Flask(__name__)

@app.route('/')
def home():
     print('Welcome To CGPA Predictor')
     return render_template('home.html')


@app.route('/pred',methods=['POST'])
def get_pred():
    print('**'*10,'initializing','**'*10)
    print('we are in pred fun of pred api')
    print('we are in post method')
    ip=request.form
    GRE=eval(ip['GRE'])
    TOEFL=eval(ip['TOEFL'])
    Rating=eval(ip['Rating'])
    SOP=eval(ip['SOP'])
    LOR=eval(ip['LOR'])
    Admit=eval(ip['Admit'])


    test_array = np.zeros(6)
    test_array[0] = GRE
    test_array[1] = TOEFL
    test_array[2] = Rating
    test_array[3] = SOP
    test_array[4] = LOR
    test_array[5] = Admit


    print('Test Array :', test_array)

        # print(f'GRE : {GRE},TOEFL : {TOEFL},Rating : {Rating},SOP : {SOP},LOR : {LOR},Admit : {Admit}')
    predicted_cgpa = np.around(l_model.predict([test_array])[0],2)

    return render_template('homepage.html',cgpa=predicted_cgpa)



if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000, debug=False)