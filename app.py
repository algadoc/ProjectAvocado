import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle   
import joblib


app = Flask(__name__)
model = joblib.load('model.pkl')
encoder = joblib.load('enc_joblib.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    print(features)
    integer_features = np.array([int(x) for x in features[:-2]])
    print([features[-2:]])
    onehots = encoder.transform(np.array(features[-2:]).reshape(-1,2)).toarray()
    print(onehots)
    print(onehots.shape)
    final_features = np.append(integer_features, onehots).reshape(-1,1)
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)