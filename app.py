from flask import Flask, render_template, request
import pandas as pd
from search import filter_and_search

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
        # pass list of dicts to template
        return render_template('results.html', results=results.to_dict(orient='records'))
    # GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

