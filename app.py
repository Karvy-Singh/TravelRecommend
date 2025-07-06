from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import numpy as np

from search import filter_and_search
from getimg import fetch_google_images
from metrics import (
    ndcg_at_k,
    precision_recall_at_k,
    average_precision_at_k,
    f_measure_at_k
)

app = FastAPI()

# Static & template setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load data
DATA_FILE = './famous_indian_tourist_places_3000.jsonl'
df_master = pd.read_json(DATA_FILE, lines=True)
df_master['id'] = df_master.index


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/results")
async def results_post(
    request: Request,
    city: str = Form(""),
    place: str = Form(""),
    rating_min: str = Form(""),
    duration_max: str = Form(""),
    month: str = Form(""),
    keywords: str = Form(""),
    top_n: str = Form("10"),
    apply_filters: Optional[str] = Form(None)
):
    uq = {
        "city": city.strip(),
        "place": place.strip(),
        "rating_min": rating_min.strip(),
        "duration_max": duration_max.strip(),
        "month": month.strip(),
        "keywords": keywords.strip(),
        "top_n": top_n.strip(),
        "apply_filters": "1" if apply_filters == "on" else "0"
    }
    qs = urlencode({k: v for k, v in uq.items() if v})
    return RedirectResponse(url=f"/results?{qs}", status_code=303)


@app.get("/results", response_class=HTMLResponse)
async def results_get(
    request: Request,
    city: str = "",
    place: str = "",
    rating_min: str = "",
    duration_max: str = "",
    month: str = "",
    keywords: str = "",
    top_n: str = "10",
    apply_filters: Optional[str] = Query(None)
):
    uq = {
        "city": city,
        "place": place,
        "rating_min": rating_min,
        "duration_max": duration_max,
        "month": month,
        "keywords": keywords,
        "top_n": top_n
    }

    apply = apply_filters == "1"

    # Run your IR/ML pipeline
    results_df = filter_and_search(df_master, uq, apply)

    # Compute metrics
    K = len(results_df)
    y_true = results_df["ratings_place"].astype(float).values
    y_rel = (y_true >= 4.0).astype(int)
    order = np.argsort(results_df["ir_score"].values)[::-1]
    y_rel_sorted = y_rel[order]
    y_grad_sorted = y_true[order]
    y_ir = results_df["ir_score"].astype(float).values
    y_ml = results_df["ml_score"].astype(float).values

    metrics = {
        "K": K,
        "ir_ndcg": ndcg_at_k(y_rel_sorted, y_ir, K),
        "ir_precision": precision_recall_at_k(y_grad_sorted, y_ir, K)[0],
        "ir_recall": precision_recall_at_k(y_grad_sorted, y_ir, K)[1],
        "ir_map": average_precision_at_k(y_grad_sorted, y_ir, K),
        "ir_F_measure": f_measure_at_k(y_grad_sorted, y_ir, K),
        "ml_ndcg": ndcg_at_k(y_true, y_ml, K),
        "ml_precision": precision_recall_at_k(y_true, y_ml, K)[0],
        "ml_recall": precision_recall_at_k(y_true, y_ml, K)[1],
        "ml_map": average_precision_at_k(y_true, y_ml, K),
        "ml_F_measure": f_measure_at_k(y_true, y_ml, K)
    }

    results = results_df.to_dict(orient="records")
    return templates.TemplateResponse("results.html", {
        "request": request,
        "results": results,
        "metrics": metrics,
        "query_args": request.query_params
    })


@app.get("/detail/{result_id}", response_class=HTMLResponse)
async def detail(request: Request, result_id: int):
    row = df_master[df_master['id'] == result_id]
    if row.empty:
        return RedirectResponse(url="/results", status_code=302)

    result = row.squeeze().to_dict()
    place = f"{result['city']},{result['place']}"
    images = fetch_google_images(place, limit=5)

    return templates.TemplateResponse("details.html", {
        "request": request,
        "result": result,
        "images": images,
        "query_args": request.query_params
    })

