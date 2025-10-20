from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        sqft_living = float(request.form['sqft_living'])
        floors = float(request.form['floors'])
        grade = float(request.form['grade'])

        # Prepare features for model
        features = np.array([[bedrooms, bathrooms, sqft_living, floors, grade]])
        prediction = model.predict(features)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted House Price: ${prediction:,.2f}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run()

