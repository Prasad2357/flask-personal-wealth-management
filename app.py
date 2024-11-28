from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models and encoders
model_savings = joblib.load('desired_savings_model.pkl')
model_percentage = joblib.load('desired_savings_percentage_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder_occupation = joblib.load('label_encoder_occupation.pkl')
label_encoder_city_tier = joblib.load('label_encoder_city_tier.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form.to_dict()
    print(f"Raw Input Data: {data}")  # Log raw data to debug

    try:
        # Convert string inputs to numeric where necessary
        data['Income'] = float(data['Income'])
        data['Age'] = int(data['Age'])
        data['Dependents'] = int(data['Dependents'])
        data['Disposable_Income'] = float(data['Disposable_Income'])
        data['Total_Expenses'] = float(data['Total_Expenses'])

        # Encode categorical data
        data['Occupation'] = label_encoder_occupation.transform([data['Occupation']])[0]
        data['City_Tier'] = label_encoder_city_tier.transform([data['City_Tier']])[0]

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all features are present in the correct order
        feature_order = ['Income', 'Age', 'Dependents', 'Occupation', 'City_Tier', 'Disposable_Income', 'Total_Expenses']
        for feature in feature_order:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing features with default values
        
        # Scale numerical features
        numerical_features = ['Income', 'Age', 'Disposable_Income', 'Total_Expenses']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])

        # Log scaled data for debugging
        print(f"DataFrame After Scaling: {input_df}")

        # Reorder columns to match model's expected order
        input_features = input_df[feature_order]

        # Make predictions
        predicted_savings = model_savings.predict(input_features)[0]
        predicted_percentage = model_percentage.predict(input_features)[0]

        # Use KMeans model to predict the risk cluster
        predicted_cluster = kmeans_model.predict(input_features)[0]
        
        # Map the predicted cluster label to a risk category
        risk_category = ""
        if predicted_cluster == 0:
            risk_category = "Low Risk"
        elif predicted_cluster == 1:
            risk_category = "Moderate Risk"
        elif predicted_cluster == 2:
            risk_category = "High Risk"
        else:
            risk_category = "Uncategorized Risk"

        # Return predictions as JSON response
        return jsonify({
            "Predicted_Savings": predicted_savings,
            "Predicted_Percentage": predicted_percentage,
            "Risk_Category": risk_category
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/goal', methods=['POST'])
def calculate_goal():
    data = request.get_json() if request.is_json else request.form.to_dict()
    try:
        # Input fields: Target amount, timeline in months, predicted savings
        target_amount = float(data['target_amount'])
        timeline_months = int(data['timeline_months'])
        predicted_savings = float(data['predicted_savings'])
        risk_category = data.get('risk_category', 'Moderate Risk')

        # Calculate required monthly savings
        required_monthly_savings = target_amount / timeline_months

        # Check if the user meets the savings goal
        savings_gap = required_monthly_savings - predicted_savings

        # Recommendations based on gap
        recommendations = []
        term_type = ""

        # Classify investment goals based on the timeline
        if timeline_months <= 12:
            term_type = "Short-Term"
        elif 12 < timeline_months <= 60:
            term_type = "Mid-Term"
        else:
            term_type = "Long-Term"

        # Recommendations for savings gap
        if savings_gap > 0:
            recommendations.append(f"Reduce discretionary expenses by ₹{savings_gap:.2f} per month.")
            recommendations.append("Consider increasing savings to close the gap.")
        else:
            recommendations.append("You are on track to meet your goal. Maintain your current savings!")

        # Investment suggestions tailored to risk category and timeline
        if term_type == "Short-Term":
            if risk_category == "Low Risk":
                recommendations.append("Invest in Fixed Deposits, Treasury Bills, or Liquid Funds for secure, short-term returns.")
            elif risk_category == "Moderate Risk":
                recommendations.append("Allocate funds to Ultra Short-Term Debt Funds or Conservative Hybrid Funds for moderate growth.")
            elif risk_category == "High Risk":
                recommendations.append("Explore Arbitrage Funds or Short-Term Equity Funds for higher returns with managed risk.")

        elif term_type == "Mid-Term":
            if risk_category == "Low Risk":
                recommendations.append("Opt for Debt Mutual Funds or Post Office Savings Schemes for stable returns.")
            elif risk_category == "Moderate Risk":
                recommendations.append("Invest in Balanced Mutual Funds or Index Funds to balance growth and risk.")
            elif risk_category == "High Risk":
                recommendations.append("Consider Sectoral Mutual Funds or Real Estate Investment Trusts (REITs) for potential high returns.")

        elif term_type == "Long-Term":
            if risk_category == "Low Risk":
                recommendations.append("Stick to Public Provident Fund (PPF), Sukanya Samriddhi Yojana, or Long-Term Bonds for safety.")
            elif risk_category == "Moderate Risk":
                recommendations.append("Diversify with Blue-Chip Stocks, Balanced Mutual Funds, or National Pension Scheme (NPS).")
            elif risk_category == "High Risk":
                recommendations.append("Maximize growth through Equity Mutual Funds, Small-Cap Stocks, or Cryptocurrencies.")

        recommendations.append(f"Since your goal is categorized as a {term_type} goal, ensure liquidity aligns with your needs.")

        return jsonify({
            "Required_Monthly_Savings": required_monthly_savings,
            "Savings_Gap": savings_gap,
            "Recommendations": recommendations
        })
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
