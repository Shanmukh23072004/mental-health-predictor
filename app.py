from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping dictionary for string inputs to numeric values
mapping_dict = {
    'not_at_all': 0,
    'several_days': 1,
    'more_than_half': 2,
    'nearly_every_day': 3,
    'never': 0,
    'almost_never': 1,
    'sometimes': 2,
    'fairly_often': 3,
    'very_often': 4,
    'male': 0,
    'female': 1,
    'other': 2,
    'Yes': 1,
    'No': 0
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Compute additional features
        sum_feature1 = sum(mapping_dict[data[key]] for key in [
            'little_interest', 'feeling_down', 'sleep_trouble', 
            'feeling_tired', 'appetite', 'feeling_failure', 
            'concentration', 'restlessness', 'anxiety'
        ])
        
        sum_feature2 = sum(mapping_dict[data[key]] for key in [
            'upset_academic', 'control_academic', 'nervous_academic', 
            'cope_academic', 'confident_academic', 'manage_stress', 
            'lonely_academic', 'high_expectations', 
            'overwhelmed_academic', 'competent_academic'
        ])
        
        sum_feature3 = sum(mapping_dict[data[key]] for key in [
            'nervous_pressure', 'stop_worrying', 'trouble_relaxing', 
            'annoyed_pressure', 'worried_academic', 'restless_pressure', 
            'overwhelmed_responsibilities'
        ])

        # Prepare the input features
        features = [
            int(data['age']),
            float(data['cgpa']),
            int(data['year_of_study']),
            mapping_dict[data['waiver']],
            *[mapping_dict[data[key]] for key in [
                'little_interest', 'feeling_down', 'sleep_trouble', 
                'feeling_tired', 'appetite', 'feeling_failure', 
                'concentration', 'restlessness', 'anxiety'
            ]],
            sum_feature1,
            *[mapping_dict[data[key]] for key in [
                'upset_academic', 'control_academic', 'nervous_academic', 
                'cope_academic', 'confident_academic', 'manage_stress', 
                'lonely_academic', 'high_expectations', 
                'overwhelmed_academic', 'competent_academic'
            ]],
            sum_feature2,
            *[mapping_dict[data[key]] for key in [
                'nervous_pressure', 'stop_worrying', 'trouble_relaxing', 
                'annoyed_pressure', 'worried_academic', 'restless_pressure', 
                'overwhelmed_responsibilities'
            ]],
            sum_feature3
        ]

        # ❌ You were wrongly using input_features (which was never created)
        # ✅ Correct it to features
        input_data = np.array([features])
        prediction = model.predict(input_data)
        prediction_class = prediction[0]

        # Map numeric prediction to mental health class
        if prediction_class <= 25:
            mental_health_status = "Healthy"
            doctor_advice = "You are doing well. No need for a doctor."
        elif prediction_class <= 50:
            mental_health_status = "Mild"
            doctor_advice = "You may want to observe your mental health and consider some lifestyle adjustments."
        elif prediction_class <= 75:
            mental_health_status = "Moderate"
            doctor_advice = "It's advisable to seek counseling or professional help."
        else:
            mental_health_status = "Severe"
            doctor_advice = "Immediate professional help is recommended. Consider consulting a doctor or mental health professional."

        return render_template('result.html', prediction=mental_health_status, advice=doctor_advice, score=prediction_class)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", advice="", score="")

if __name__ == '__main__':
    app.run(debug=True)
