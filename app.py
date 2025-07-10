from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_model.pkl")

@app.route('/')
def first():  # Load 'first.html' as the homepage
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')

        if 'Id' in df.columns:  # Check if 'Id' column exists
            df.set_index('Id', inplace=True)

        df_html = df.to_html(classes="table table-striped", na_rep="-")  # Convert to HTML
        return render_template("preview.html", df_view=df_html)  # Pass as a string

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form
    input_data = {
        "trans_hour": int(form_data['trans_hour']),
        "trans_day": int(form_data['trans_day']),
        "trans_month": int(form_data['trans_month']),
        "trans_year": int(form_data['trans_year']),
        "trans_amount": float(form_data['trans_amount']),
        "upi_number": form_data['upi_number']  # Add UPI number
    }

    # Create a DataFrame from input
    df = pd.DataFrame([input_data])

    # Predict with the model
    prediction = model.predict(df)[0]

    # Convert prediction to label
    result = "Fraud" if prediction == 1 else "Not Fraud"

    return render_template('index.html', result=result)

@app.route('/prediction1', methods=['GET'])
def prediction1():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    app.run(debug=True)
