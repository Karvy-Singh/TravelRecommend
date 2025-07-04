# app.py
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from search import filter_and_search
from getimg import fetch_wiki_images
from metrics import (
    ndcg_at_k,
    precision_recall_at_k,
    average_precision_at_k,
    f_measure_at_k
)

DATA_FILE = './famous_indian_tourist_places_3000.jsonl'
app = Flask(__name__)

df_master = pd.read_json(DATA_FILE, lines=True)
df_master['id'] = df_master.index

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        # Collect form inputs
        apply_filters = request.form.get('apply_filters') == 'on'
        uq = {
            'city':         request.form.get('city', '').strip(),
            'place':        request.form.get('place', '').strip(),
            'rating_min':   request.form.get('rating_min', '').strip(),
            'duration_max': request.form.get('duration_max', '').strip(),
            'month':        request.form.get('month', '').strip(),
            'keywords':     request.form.get('keywords', '').strip(),
            'top_n':        request.form.get('top_n', '10').strip(),
            'apply_filters': '1' if apply_filters else '0'
        }
        # Redirect-to-GET with all params in the query string
        qs = urlencode({k: v for k, v in uq.items() if v})
        return redirect(f"{url_for('results')}?{qs}")

    apply_filters = request.args.get('apply_filters') == '1'
    uq = {
        'city':         request.args.get('city', ''),
        'place':        request.args.get('place', ''),
        'rating_min':   request.args.get('rating_min', ''),
        'duration_max': request.args.get('duration_max', ''),
        'month':        request.args.get('month', ''),
        'keywords':     request.args.get('keywords', ''),
        'top_n':        request.args.get('top_n', '10')
    }

    # run your IR/ML pipeline
    results_df = filter_and_search(df_master, uq, apply_filters)

    # compute metrics
    K      = len(results_df)
    y_true = results_df["ratings_place"].astype(float).values
    y_rel  = (y_true >= 4.0).astype(int)
    order = np.argsort(results_df["ir_score"].values)[::-1]
    y_rel_sorted = y_rel[order]
    y_grad_sorted = y_true[order]
    y_ir   = results_df["ir_score"].astype(float).values
    y_ml   = results_df["ml_score"].astype(float).values

    metrics = {
        "K":            K,
        "ir_ndcg":      ndcg_at_k(y_rel_sorted, y_ir, K),
        "ir_precision": precision_recall_at_k(y_grad_sorted, y_ir, K)[0],
        "ir_recall":    precision_recall_at_k(y_grad_sorted, y_ir, K)[1],
        "ir_map":       average_precision_at_k(y_grad_sorted, y_ir, K),
        "ir_F_measure": f_measure_at_k(y_grad_sorted, y_ir, K),
        "ml_ndcg":      ndcg_at_k(y_true, y_ml, K),
        "ml_precision": precision_recall_at_k(y_true, y_ml, K)[0],
        "ml_recall":    precision_recall_at_k(y_true, y_ml, K)[1],
        "ml_map":       average_precision_at_k(y_true, y_ml, K),
        "ml_F_measure": f_measure_at_k(y_true, y_ml, K)
    }

    # hand off to your template
    results = results_df.to_dict(orient='records')
    return render_template('results.html',
                           results=results,
                           metrics=metrics,
                           query_args=request.args)


@app.route('/detail/<int:result_id>')
def detail(result_id):
    row = df_master[df_master['id'] == result_id]
    if row.empty:
        return redirect(url_for('results'))
    result = row.squeeze().to_dict()
    images = []
    place = result["city"]
    images = fetch_wiki_images(place, limit=5)
    print(images)
    # pass along the original query args so the "Back" link can restore them
    return render_template('details.html',
                           result=result,
                           query_args=request.args,
                           images=images)


if __name__ == '__main__':
    app.run(debug=True)

