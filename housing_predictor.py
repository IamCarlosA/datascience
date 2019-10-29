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
	"""
	CRIM = content['CRIM']
	ZN = content['	ZN']
	INDUS = content['INDUS'] 
	CHAS = content['CHAS']  
	NOX = content['NOX']    
	RM = content['RM']   
	AGE = content['AGE']    
	DIS = content['DIS']
	RAD = content['RAD']   
	TAX = content['TAX']
	PTRATIO = content['PTRATIO']       
	B = content['B'] 
	LSTAT = content['LSTAT']

	"""
	CRIM = 0.00632
	ZN = 8.0  
	INDUS = 2.31  
	CHAS = 1    
	NOX = 0.538     
	RM = 6.575   
	AGE = 65.2    
	DIS = 1.0900 
	RAD = 1    
	TAX = 296.0
	PTRATIO = 15.3       
	B = 26.90  
	LSTAT = 4.98  
	

	new_df = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
	# We scale those values like the others
	new_df_scaled = scaler.transform(new_df)

	# We predict the outcome
	prediction = svc.predict(new_df_scaled)
	
	diagnostic = prediction[0]

	return jsonify({'housing ':str(diagnostic)})




if __name__ == '__main__':
    app.run()