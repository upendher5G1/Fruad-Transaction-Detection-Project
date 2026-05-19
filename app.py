from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load ML Model
model = pickle.load(open("fraud_model.pkl", "rb"))

# Create Database
def init_db():
    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL,
        oldbalance REAL,
        newbalance REAL,
        transaction_type TEXT,
        prediction TEXT,
        risk_score REAL,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        amount = float(request.form["amount"])
        oldbalance = float(request.form["oldbalance"])
        newbalance = float(request.form["newbalance"])
        transaction_type = request.form["transaction_type"]

        # Convert transaction type
        type_value = 1 if transaction_type == "TRANSFER" else 0

        # Features
        features = np.array([[amount, oldbalance]])

        # Prediction
        prediction = model.predict(features)[0]

        # Probability
        probability = model.predict_proba(features)[0][1]

        risk_score = round(probability * 100, 2)

        if prediction == 1:
            result = "Fraud Transaction Detected"
        else:
            result = "Normal Transaction"

        # Save to DB
        conn = sqlite3.connect("transactions.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO transactions
        (amount, oldbalance, newbalance, transaction_type,
         prediction, risk_score, time)

        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            amount,
            oldbalance,
            newbalance,
            transaction_type,
            result,
            risk_score,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return render_template(
            "index.html",
            prediction_text=result,
            risk_score=risk_score
        )

    except Exception as e:
        return f"Error: {e}"


@app.route("/history")
def history():

    conn = sqlite3.connect("transactions.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM transactions ORDER BY id DESC")

    data = cursor.fetchall()

    conn.close()

    return render_template("history.html", transactions=data)


if __name__ == "__main__":
    app.run(debug=True)