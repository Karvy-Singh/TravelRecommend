from flask import Flask, render_template, request
import pandas as pd
from search import filter_and_search
from metrics import (
    ndcg_at_k,
    precision_recall_at_k,
    average_precision_at_k,
    f_measure_at_k
)

DATA_FILE = './famous_indian_tourist_places_3000.jsonl'
app = Flask(__name__)

# load once at startup
df_master = pd.read_json(DATA_FILE, lines=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # grab form values
        apply_filters = (request.form.get('apply_filters') == 'on')
        uq = {
            'city'        : request.form.get('city', '').strip(),
            'place'       : request.form.get('place', '').strip(),
            'rating_min'  : request.form.get('rating_min', '').strip(),
            'duration_max': request.form.get('duration_max', '').strip(),
            'month'       : request.form.get('month', '').strip(),
            'keywords'    : request.form.get('keywords', '').strip(),
            'top_n'       : request.form.get('top_n', '10').strip()
        }
        results = filter_and_search(df_master, uq, apply_filters)
        K      = len(results)
        y_true = results["ratings_place"].astype(float).values
        y_ir   = results["ir_score"].astype(float).values
        y_ml   = results["ml_score"].astype(float).values

        metrics = {
            "K":       K,
            "ir_ndcg": ndcg_at_k(y_true, y_ir, K),
            "ir_precision":    precision_recall_at_k(y_true, y_ir, K)[0],
            "ir_recall": precision_recall_at_k(y_true, y_ir, K)[1],
            "ir_map":  average_precision_at_k(y_true, y_ir, K),
            "ir_F_measure": f_measure_at_k(y_true, y_ir, K),
            "ml_ndcg": ndcg_at_k(y_true, y_ml, K),
            "ml_precision":    precision_recall_at_k(y_true, y_ml, K)[0],
            "ml_recall": precision_recall_at_k(y_true, y_ml, K)[1],
            "ml_map":  average_precision_at_k(y_true, y_ml, K),
            "ml_F_measure": f_measure_at_k(y_true, y_ml, K)
        }
        # pass list of dicts to template
        return render_template('results.html', results=results.to_dict(orient='records'),metrics=metrics)
    # GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

