
import os, warnings, joblib, pandas as pd
from flask import Flask, render_template, request
from config.paths_config import MODEL_OUTPUT_PATH

FEATURES = [
    "lead_time", "no_of_special_request", "avg_price_per_room",
    "arrival_month", "arrival_date", "market_segment_type",
    "no_of_week_nights", "no_of_weekend_nights",
    "type_of_meal_plan", "room_type_reserved"
]

app = Flask(__name__)
loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # collect and cast user inputs
        vals = [
            int(request.form.get("lead_time", 0)),
            int(request.form.get("no_of_special_request", 0)),
            float(request.form.get("avg_price_per_room", 0)),
            int(request.form.get("arrival_month", 0)),
            int(request.form.get("arrival_date", 0)),
            int(request.form.get("market_segment_type", 0)),
            int(request.form.get("no_of_week_nights", 0)),
            int(request.form.get("no_of_weekend_nights", 0)),
            int(request.form.get("type_of_meal_plan", 0)),
            int(request.form.get("room_type_reserved", 0)),
        ]

        X = pd.DataFrame([vals], columns=FEATURES)
        
        keep_prob = loaded_model.predict_proba(X)[0][1]
        label = int(keep_prob >= 0.5)  

        return render_template(
            "index.html",
            prediction=label,
            probability=keep_prob
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
