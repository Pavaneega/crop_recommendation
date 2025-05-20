from flask import Flask, request, render_template
import joblib
import numpy as np
import os


# Get the current file's directory
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'knn_model.pkl')

app = Flask(__name__)

# Load the model
model = joblib.load(model_path)






@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract user inputs from the form
            N = float(request.form['N'])
            P = float(request.form['P'])
            k = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])

            # Predict the crop
            features = np.array([[N, P, k, temperature, humidity, ph, rainfall]])
            pred = model.predict(features)[0]

            # Map prediction to crop names
            crops = [
                "Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", "Orange",
                "Apple", "Muskmelon", "Watermelon", "Grapes", "Mango", "Banana",
                "Pomegranate", "Lentil", "Blackgram", "Mungbean", "Mothbeans",
                "Pigeonpeas", "Kidneybeans", "Chickpea", "Coffee"
            ]

            target = crops[pred - 1] if 1 <= pred <= len(crops) else "Sorry, we are not able to recommend a proper crop for this environment"

        except ValueError as e:
            target = "Invalid input. Please make sure all inputs are numbers."

        return render_template('index.html', result=target)

    return render_template('index.html', result='')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
