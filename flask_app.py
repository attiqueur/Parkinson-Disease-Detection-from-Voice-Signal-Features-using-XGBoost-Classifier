from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('xgb_parkinson_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html',
                           title='Parkinson Disease Detection from Voice Signal Features using XGBoost Classifier')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input features from the form and convert them to float
        MDVP_Fo = float(request.form['MDVP_Fo'])
        MDVP_Fhi = float(request.form['MDVP_Fhi'])
        MDVP_Flo = float(request.form['MDVP_Flo'])
        MDVP_Jitter = float(request.form['MDVP_Jitter'])
        MDVP_Shimmer = float(request.form['MDVP_Shimmer'])
        NHR = float(request.form['NHR'])
        HNR = float(request.form['HNR'])
        RPDE = float(request.form['RPDE'])
        DFA = float(request.form['DFA'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        D2 = float(request.form['D2'])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR,
                                    RPDE, DFA, spread1, spread2, D2]],
                                  columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                           'MDVP:Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2'])

        # Make prediction
        prediction = model.predict(input_data)

        # Map prediction to human-readable label
        if prediction[0] == 0:
            result = 'Normal'
        else:
            result = 'Parkinson'

        # Render the template with prediction result included
        return render_template('index.html', title='Parkinson Prediction Result', result=result)


if __name__ == '__main__':
    app.run(debug=True)
