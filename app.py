# main.py
from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
from urllib.parse import urlencode
from typing import Optional

from search import filter_and_search
from getimg import fetch_google_images
from metrics import (
    ndcg_at_k,
    precision_recall_at_k,
    average_precision_at_k,
    f_measure_at_k
)

DATA_FILE = './famous_indian_tourist_places_3000.jsonl'

app = FastAPI()

# serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# load once
df_master = pd.read_json(DATA_FILE, lines=True)
df_master['id'] = df_master.index


@app.get("/", name="index")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/results", name="results_post")
async def results_post(request: Request):
    # build your query‐dict exactly as before
    form = await request.form()
    print(form)
    apply_filters = form.get('apply_filters') == 'on'
    uq = {
            'city':         form.get('city', '').strip(),
            'place':        form.get('place', '').strip(),
            'rating_min':   form.get('rating_min', '').strip(),
            'duration_max': form.get('duration_max', '').strip(),
            'month':        form.get('month', '').strip(),
            'keywords':     form.get('keywords', '').strip(),
            'top_n':        form.get('top_n', '10').strip(),
            'apply_filters': '1' if apply_filters else '0'
    }
    # only include non‐empty
    qs = urlencode({k: v for k, v in uq.items() if v})
    # redirect to GET /results
    url = str(request.url_for("results_get")) + "?" + qs
    return RedirectResponse(url=url, status_code=303)


@app.get("/results", name="results_get")
async def results_get(request: Request):
    qp = request.query_params
    apply_filters = qp.get("apply_filters") == "1"
    uq = {
        "city":         qp.get("city", ""),
        "place":        qp.get("place", ""),
        "rating_min":   qp.get("rating_min", ""),
        "duration_max": qp.get("duration_max", ""),
        "month":        qp.get("month", ""),
        "keywords":     qp.get("keywords", ""),
        "top_n":        qp.get("top_n", "10"),
    }

    # IR/ML pipeline
    results_df = filter_and_search(df_master, uq, apply_filters)

    # metrics
    K      = len(results_df)
    y_true = results_df["ratings_place"].astype(float).values
    y_rel  = (y_true >= 4.0).astype(int)
    order  = np.argsort(results_df["ir_score"].values)[::-1]
    y_rel_sorted  = y_rel[order]
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
        "ml_F_measure": f_measure_at_k(y_true, y_ml, K),
    }

    return templates.TemplateResponse(
        "results.html",
        {
            "request":    request,
            "results":    results_df.to_dict(orient="records"),
            "metrics":    metrics,
            "query_args": qp
        }
    )


@app.get("/detail/{result_id}", name="detail")
async def detail(request: Request, result_id: int):
    row = df_master[df_master["id"] == result_id]
    if row.empty:
        # back to GET /results
        return RedirectResponse(request.url_for("results_get"))
    result = row.squeeze().to_dict()
    place  = f"{result['city']},{result['place']}"
    images = fetch_google_images(place, limit=5)

    return templates.TemplateResponse(
        "details.html",
        {
            "request":    request,
            "result":     result,
            "query_args": request.query_params,
            "images":     images
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
