#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        temp_C = float(request.form['temp_C'])
        dew_point_temp_C = float(request.form['dew_point_temp_C'])
        rel_hum = float(request.form['rel_hum'])
        wind_speed_km_h = float(request.form['wind_speed_km_h'])
        visibility_km = float(request.form['visibility_km'])
        press_kPa = float(request.form['press_kPa'])

        # Make prediction using the loaded model
        prediction = model.predict([[temp_C, dew_point_temp_C, rel_hum, wind_speed_km_h, visibility_km, press_kPa]])

        # Pass the prediction back to the user interface
        return render_template('index.html', prediction=prediction[0])
        
if __name__ == '__main__':
    app.run()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script we.ipynb')

