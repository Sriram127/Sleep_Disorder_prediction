from flask import Flask, request, jsonify, render_template, session
import secrets
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import io
import base64
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(32)
app.config['SESSION_PERMANENT'] = False  # Ensure sessions are not persistent

# Load the saved models, encoder, and scaler
model_rf = pickle.load(open(r"best_student_model.sav", 'rb'))  
encoder = pickle.load(open(r"student_encoder.sav", 'rb')) 
scaler = pickle.load(open(r"student_scaler.sav", 'rb')) 

# Load the original dataset for graph data
data = pd.read_csv(r"Student_Performance_.csv")
data["Extracurricular Activities"] = encoder.transform(data["Extracurricular Activities"])

# Load pre-trained models
with open(r"trained_decision_tree_model.sav", 'rb') as file: 
    model_dt = pickle.load(file)

with open(r"trained_linear_regression_model.sav", 'rb') as file: 
    model_lr = pickle.load(file)

with open(r"trained_svm_model.sav", 'rb') as file: 
    model_svm = pickle.load(file)

# Function to generate a graph image
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
def generate_graph(model_name, x_data, y_data, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, 'o-')  
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Create a buffer for the image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Encode the image in base64
    graph_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return graph_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data directly using the correct names
        hours_studied = int(request.form.get('hours_studied', 0))  # Handle missing values
        previous_scores = int(request.form.get('previous_scores', 0))  # Handle missing values
        extracurricular_activities = request.form.get('extracurricular_activities', 'No')  # Handle missing values
        sleep_hours = int(request.form.get('sleep_hours', 0))  # Handle missing values
        sample_papers = int(request.form.get('sample_question_papers_practiced', 0))  # Handle missing values
        ml_midterm = int(request.form.get('ML Mid term exam', 0))  # Handle missing values
        es_midterm = int(request.form.get('ES Mid term exam', 0))  # Handle missing values
        cc_midterm = int(request.form.get('CC Mid term exam', 0))  # Handle missing values
        cn_midterm = int(request.form.get('CN Mid term exam', 0))  # Handle missing values
        bda_midterm = int(request.form.get('BDA Mid term exam', 0))  # Handle missing values

        # Create a dictionary with the correct keys
        new_data = {
            'Hours Studied': hours_studied,
            'Previous Scores': previous_scores,
            'Sleep Hours': sleep_hours,
            'Sample Question Papers Practiced': sample_papers,
            'ML Mid term exam': ml_midterm,
            'ES Mid term exam': es_midterm,
            'CC Mid term exam': cc_midterm,
            'CN Mid term exam': cn_midterm,
            'BDA Mid term exam': bda_midterm
        }

        # Transform the 'Extracurricular Activities' value
        extracurricular_activities_encoded = encoder.transform([extracurricular_activities])[0] 
        new_data['Extracurricular Activities'] = extracurricular_activities_encoded

        # Create the input DataFrame with the correct column order
        input_df = pd.DataFrame([new_data], columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'ML Mid term exam', 'ES Mid term exam', 'CC Mid term exam', 'CN Mid term exam', 'BDA Mid term exam'])
        transformed_input = scaler.transform(input_df)

        prediction = model_rf.predict(transformed_input)[0]

        # Store user input in the session
        session['hours_studied'] = hours_studied
        session['previous_scores'] = previous_scores
        session['extracurricular_activities'] = extracurricular_activities
        session['sleep_hours'] = sleep_hours
        session['sample_papers'] = sample_papers
        session['ml_midterm'] = ml_midterm
        session['es_midterm'] = es_midterm
        session['cc_midterm'] = cc_midterm
        session['cn_midterm'] = cn_midterm
        session['bda_midterm'] = bda_midterm

        return render_template('predict.html', prediction=prediction)
    else:
        return render_template('index.html')
@app.route('/graphs')
def graphs():
    # Retrieve user input from the session
    hours_studied = session.get('hours_studied', 0)
    previous_scores = session.get('previous_scores', 0)
    extracurricular_activities = session.get('extracurricular_activities', 'No')
    sleep_hours = session.get('sleep_hours', 0)
    sample_papers = session.get('sample_papers', 0)
    ml_midterm = session.get('ml_midterm', 0)
    es_midterm = session.get('es_midterm', 0)
    cc_midterm = session.get('cc_midterm', 0)
    cn_midterm = session.get('cn_midterm', 0)
    bda_midterm = session.get('bda_midterm', 0)

    print("User Input:")
    print(f"Hours Studied: {hours_studied}, Previous Scores: {previous_scores}, Extracurricular Activities: {extracurricular_activities}")
    print(f"Sleep Hours: {sleep_hours}, Sample Papers: {sample_papers}")
    print(f"Midterm Scores - ML: {ml_midterm}, ES: {es_midterm}, CC: {cc_midterm}, CN: {cn_midterm}, BDA: {bda_midterm}")

    # Filter data based on user input
    try:
        filtered_data = data[
            (data['Hours Studied'] >= hours_studied) &
            (data['Previous Scores'] >= previous_scores) &
            (data['Extracurricular Activities'] == encoder.transform([extracurricular_activities])[0]) &
            (data['Sleep Hours'] >= sleep_hours) &
            (data['Sample Question Papers Practiced'] >= sample_papers) &
            (data['ML Mid term exam'] >= ml_midterm) &
            (data['ES Mid term exam'] >= es_midterm) &
            (data['CC Mid term exam'] >= cc_midterm) &
            (data['CN Mid term exam'] >= cn_midterm) &
            (data['BDA Mid term exam'] >= bda_midterm)
        ]
    except Exception as e:
        print(f"Error during data filtering: {e}")
        filtered_data = pd.DataFrame()  # Empty DataFrame

    print("Filtered Data:")
    print(filtered_data)

    if filtered_data.empty:
        graph_data_rf = "No data available for this filter"
        graph_data_dt = "No data available for this filter"
        graph_data_lr = "No data available for this filter"
        graph_data_svm = "No data available for this filter"
    else:
        try:
            # Generate predictions for each model
            rf_predictions = model_rf.predict(filtered_data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'ML Mid term exam', 'ES Mid term exam', 'CC Mid term exam', 'CN Mid term exam', 'BDA Mid term exam']])
            print("RF Predictions:")
            print(rf_predictions)
            graph_data_rf = generate_graph(
                'Random Forest', filtered_data['Hours Studied'], 
                rf_predictions,
                'Performance vs Hours Studied (Random Forest)', 'Hours Studied', 'Performance Index'
            )
        except Exception as e:
            print(f"Error during RF predictions or graph generation: {e}")
            graph_data_rf = "Error generating graph"

        try:
            dt_predictions = model_dt.predict(filtered_data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'ML Mid term exam', 'ES Mid term exam', 'CC Mid term exam', 'CN Mid term exam', 'BDA Mid term exam']])
            print("DT Predictions:")
            print(dt_predictions)
            graph_data_dt = generate_graph(
                'Decision Tree', filtered_data['Sleep Hours'], 
                dt_predictions,
                'Performance vs Sleep Hours (Decision Tree)', 'Sleep Hours', 'Performance Index'
            )
        except Exception as e:
            print(f"Error during DT predictions or graph generation: {e}")
            graph_data_dt = "Error generating graph"

        try:
            lr_predictions = model_lr.predict(filtered_data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'ML Mid term exam', 'ES Mid term exam', 'CC Mid term exam', 'CN Mid term exam', 'BDA Mid term exam']])
            print("LR Predictions:")
            print(lr_predictions)
            graph_data_lr = generate_graph(
                'Linear Regression', filtered_data['Previous Scores'], 
                lr_predictions,
                'Performance vs Previous Scores (Linear Regression)', 'Previous Scores', 'Performance Index'
            )
        except Exception as e:
            print(f"Error during LR predictions or graph generation: {e}")
            graph_data_lr = "Error generating graph"

        try:
            svm_predictions = model_svm.predict(filtered_data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced', 'ML Mid term exam', 'ES Mid term exam', 'CC Mid term exam', 'CN Mid term exam', 'BDA Mid term exam']])
            print("SVM Predictions:")
            print(svm_predictions)
            graph_data_svm = generate_graph(
                'SVM', filtered_data['Sample Question Papers Practiced'], 
                svm_predictions,
                'Performance vs Sample Papers Practiced (SVM)', 'Sample Papers', 'Performance Index'
            )
        except Exception as e:
            print(f"Error during SVM predictions or graph generation: {e}")
            graph_data_svm = "Error generating graph"

    graph_data = {
        'Random Forest': graph_data_rf,
        'Decision Tree': graph_data_dt,
        'Linear Regression': graph_data_lr,
        'SVM': graph_data_svm 
    }

    return render_template('graph.html', graph_data=graph_data)



if __name__ == '__main__':
    app.run(debug=True)