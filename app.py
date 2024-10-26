from flask import Flask, request, render_template
from model import StudentScorePredictor

app = Flask(__name__)

# Initialize the model with the path to your CSV file
predictor = StudentScorePredictor(data_path='student-mat.csv')

@app.route('/')
def home():
    # Define dropdown options
    dropdown_options = {
        'school': ['GP', 'MS'],
        'sex': ['F', 'M'],
        'address': ['U', 'R'],
        'famsize': ['GT3', 'LE3'],
        'Pstatus': ['T', 'A'],
        'Mjob': ['other', 'services', 'at_home', 'teacher', 'health'],
        'Fjob': ['other', 'services', 'teacher', 'at_home', 'health'],
        'reason': ['course', 'home', 'reputation', 'other'],
        'guardian': ['mother', 'father', 'other'],
        'schoolsup': ['no', 'yes'],
        'famsup': ['yes', 'no'],
        'paid': ['no', 'yes'],
        'activities': ['yes', 'no'],
        'nursery': ['yes', 'no'],
        'higher': ['yes', 'no'],
        'internet': ['yes', 'no'],
        'romantic': ['no', 'yes']
    }
    
    # Get feature names from the model
    features = predictor.features
    
    # Create a set of dropdown feature names for template
    dropdown_features = set(dropdown_options.keys())
    
    return render_template('index.html', features=features, dropdown_features=dropdown_features, dropdown_options=dropdown_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [
            float(request.form.get('age', 0)),
            request.form.get('school'),
            request.form.get('sex'),
            request.form.get('address'),
            request.form.get('famsize'),
            request.form.get('Pstatus'),
            request.form.get('Mjob'),
            request.form.get('Fjob'),
            request.form.get('reason'),
            request.form.get('guardian'),
            request.form.get('schoolsup'),
            request.form.get('famsup'),
            request.form.get('paid'),
            request.form.get('activities'),
            request.form.get('nursery'),
            request.form.get('higher'),
            request.form.get('internet'),
            request.form.get('romantic'),
            float(request.form.get('famrel', 0)),
            float(request.form.get('freetime', 0)),
            float(request.form.get('goout', 0)),
            float(request.form.get('Dalc', 0)),
            float(request.form.get('Walc', 0)),
            float(request.form.get('health', 0)),
            float(request.form.get('absences', 0)),
            float(request.form.get('G1', 0)),
            float(request.form.get('G2', 0))
        ]
        
        # Check input_features length
        if len(input_features) != len(predictor.features):
            return render_template('index.html', prediction_text='Error: Incorrect number of features provided.')
        
    except ValueError as e:
        return render_template('index.html', prediction_text=f'Error: Invalid input values. {e}')
    
    try:
        prediction = predictor.run(input_features)
        return render_template('index.html', prediction_text=f'Predicted score: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True,port=2020)
