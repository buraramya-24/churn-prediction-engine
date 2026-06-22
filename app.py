from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

# 1. Initialize the FastAPI App
app = FastAPI(title="Enterprise Churn Prediction API")

# 2. Load the trained AI model brains and feature structures
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

# 3. Define what data inputs the API expects (Incoming Schema)
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Gender_Male: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0

# 4. Serve the Interactive Webpage Front-End
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Churn Predictor UI</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 40px; }
            .container { max-width: 500px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin: auto; }
            h2 { color: #333; text-align: center; margin-bottom: 20px; }
            label { font-weight: bold; display: block; margin-top: 15px; color: #555; }
            select, input { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            button { width: 100%; background-color: #2ecc71; color: white; padding: 12px; border: none; border-radius: 4px; font-size: 16px; margin-top: 25px; cursor: pointer; font-weight: bold; }
            button:hover { background-color: #27ae60; }
            #result { margin-top: 25px; padding: 15px; border-radius: 4px; text-align: center; font-weight: bold; display: none; }
            .high-risk { background-color: #fadbd8; color: #78281f; border: 1px solid #f5b7b1; }
            .low-risk { background-color: #d4efdf; color: #145a32; border: 1px solid #abebc6; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🔮 Churn Risk Predictor</h2>
            <form id="churnForm">
                <label>Senior Citizen?</label>
                <select id="SeniorCitizen"><option value="0">No</option><option value="1">Yes</option></select>
                
                <label>Tenure (Months with company):</label>
                <input type="number" id="tenure" value="12" min="0">
                
                <label>Monthly Charges ($):</label>
                <input type="number" id="MonthlyCharges" value="50.00" step="0.01">
                
                <label>Total Charges ($):</label>
                <input type="number" id="TotalCharges" value="600.00" step="0.01">
                
                <label>Contract Type:</label>
                <select id="contract">
                    <option value="month">Month-to-Month</option>
                    <option value="one">One Year</option>
                    <option value="two">Two Years</option>
                </select>

                <label>Internet Service Type:</label>
                <select id="internet">
                    <option value="dsl">DSL</option>
                    <option value="fiber">Fiber Optic</option>
                    <option value="no">No Internet Service</option>
                </select>
                
                <button type="button" onclick="makePrediction()">Analyze Customer Risk</button>
            </form>
            <div id="result"></div>
        </div>

        <script>
            async function makePrediction() {
                const contractVal = document.getElementById('contract').value;
                const internetVal = document.getElementById('internet').value;
                
                const payload = {
                    SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
                    tenure: parseInt(document.getElementById('tenure').value),
                    MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                    TotalCharges: parseFloat(document.getElementById('TotalCharges').value),
                    Gender_Male: 1,
                    Contract_One_year: contractVal === 'one' ? 1 : 0,
                    Contract_Two_year: contractVal === 'two' ? 1 : 0,
                    InternetService_Fiber_optic: internetVal === 'fiber' ? 1 : 0,
                    InternetService_No: internetVal === 'no' ? 1 : 0,
                    PaymentMethod_Credit_card_automatic: 0,
                    PaymentMethod_Electronic_check: 1,
                    PaymentMethod_Mailed_check: 0
                };

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerText = `Prediction: ${data.status} (${data.churn_probability}% Probability)`;
                
                if(data.churn_prediction === 1) {
                    resultDiv.className = 'high-risk';
                } else {
                    resultDiv.className = 'low-risk';
                }
            }
        </script>
    </body>
    </html>
    """

# 5. Handle the visual form submissions
@app.post("/predict")
def predict_churn(customer: CustomerData):
    input_data = customer.model_dump()
    full_input_dict = {feat: 0 for feat in model_features}
    
    for key, value in input_data.items():
        if key in full_input_dict:
            full_input_dict[key] = value
            
    input_df = pd.DataFrame([full_input_dict])
    input_df = input_df[model_features]
    
    probability = model.predict_proba(input_df)[0][1]
    prediction = int(model.predict(input_df)[0])
    
    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(probability) * 100, 2),
        "status": "High Risk - Take Retention Action!" if prediction == 1 else "Low Risk - Stable"
    }
