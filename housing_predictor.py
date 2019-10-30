from flask import Flask, jsonify, request
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as Scaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello Housing!'

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	# load the model from disk
	filename = 'housing.model'
	svc = joblib.load(filename)

	# load the scaler from disk
	filename = 'scaler.scaler'
	scaler = joblib.load(filename)


	content = request.json
	print(content)
	
	crim = content['crim']
	zn = content['zn']
	indus = content['indus'] 
	chas = content['chas']  
	nox = content['nox']    
	rm = content['rm']   
	age = content['age']    
	dis = content['dis']
	rad = content['rad']   
	tax = content['tax']
	ptratio = content['ptratio']       
	b = content['b'] 
	lstat = content['lstat']

	new_df = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
	# We scale those values like the others
	new_df_scaled = scaler.transform(new_df)

	# We predict the outcome
	prediction = svc.predict(new_df_scaled)
	
	diagnostic = prediction[0]

	return jsonify({'PRECIO ':str(diagnostic)})




if __name__ == '__main__':
    app.run(debug=True)