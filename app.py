from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("vehicle_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        mileage = float(request.form["mileage"])
        engine_temp = float(request.form["engine_temp"])
        rpm = float(request.form["rpm"])
        oil_pressure = float(request.form["oil_pressure"])
        fuel_efficiency = float(request.form["fuel_efficiency"])

        features = np.array([[mileage, engine_temp, rpm, oil_pressure, fuel_efficiency]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            result = "⚠ Needs Maintenance!"
        else:
            result = "✅ No Maintenance Needed."

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
