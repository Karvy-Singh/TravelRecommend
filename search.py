import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from xgboost import XGBRegressor

DATA_FILE = "famous_indian_tourist_places_3000.jsonl"

MONTH_MAP = {
    m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"],
        start=1
    )
}

def parse_duration(s):
    """Return mean of '2-4' → 3.0, or float(s), else nan."""
    try:
        if isinstance(s, str) and "-" in s:
            a, b = map(float, s.split("-",1))
            return (a + b) / 2
        return float(s)
    except:
        return np.nan

def in_month_range(range_str, month_str):
    """
    Check if month_str (e.g. "May") falls within a range_str like "October-June".
    Handles wrap-around ranges.
    """
    if not isinstance(range_str, str) or not month_str:
        return False

    t = month_str.strip().lower()
    target = MONTH_MAP.get(t)
    if target is None:
        return False

    parts = [p.strip().lower() for p in range_str.split('-', 1)]
    if len(parts) == 2:
        start = MONTH_MAP.get(parts[0])
        end   = MONTH_MAP.get(parts[1])
        if start is None or end is None:
            return False
        if start <= end:
            return start <= target <= end
        else:
            # e.g. October (10) → June (6): wrap around year end
            return target >= start or target <= end
    else:
        # single month or free‐text match
        return t in range_str.lower()


# 1. Read full data
df = pd.read_json(DATA_FILE, lines=True)

# 2. Text feature
df["combined_text"] = (
    df["place_desc"].fillna("") + " " + df["city_desc"].fillna("")
).str.strip()
tfidf_global = TfidfVectorizer(stop_words="english", max_features=15_000)

tfidf_city  = TfidfVectorizer(stop_words="english", max_features=2_000)\
                  .fit(df["city"].fillna(""))
tfidf_place = TfidfVectorizer(stop_words="english", max_features=2_000)\
                  .fit(df["place"].fillna(""))
X_text_full = tfidf_global.fit_transform(df["combined_text"])
X_city_full  = tfidf_city.transform(df["city"].fillna(""))
X_place_full = tfidf_place.transform(df["place"].fillna(""))


# 3. Numeric feature: ideal duration
dur = df["ideal_duration"].apply(parse_duration).fillna(df["ideal_duration"].apply(parse_duration).mean())
dur = dur.values.reshape(-1, 1)
scaler = StandardScaler()
X_num_full = csr_matrix(scaler.fit_transform(dur))

# 4. Combine
X_full = hstack([X_text_full,X_city_full,X_place_full, X_num_full]).tocsr()
y_full = df["ratings_place"].astype(float).values

# 5. Train regressor once
reg = XGBRegressor(
    tree_method="hist",
    max_bin=128,
    n_estimators=200,
    learning_rate=0.15,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4
)
reg.fit(X_full, y_full)

from sklearn.feature_extraction.text import TfidfVectorizer

def filter_and_search(df: pd.DataFrame, uq: dict, apply_filters: bool):
    df = df.copy()
    df['combined_text'] = (
        df['place_desc'].fillna('') + ' ' + df['city_desc'].fillna('')
    ).str.strip()

    # Hard filters
    if apply_filters:
        if uq.get('city'):
            df = df[df['city'].str.contains(uq['city'], case=False, na=False)]
        if uq.get('place'):
            df = df[df['place'].str.contains(uq['place'], case=False, na=False)]
        if uq.get('rating_min'):
            try:
                min_r = float(uq['rating_min'])
                df = df[df['ratings_place'] >= min_r]
            except ValueError:
                pass
        if uq.get('duration_max'):
            try:
                max_d = int(uq['duration_max'])
                df = df[df['ideal_duration']
                        .apply(lambda s: (parse_duration(s)[1] is not None)
                                       and (parse_duration(s)[1] <= max_d))]
            except ValueError:
                pass
        if uq.get('month'):
            df = df[df['best_time_to_visit']
                    .fillna('')
                    .apply(lambda rng: in_month_range(rng, uq['month']))]

    # IR step
    if uq.get('keywords') and not df.empty:
        vec = TfidfVectorizer(stop_words='english')
        M   = vec.fit_transform(df['combined_text'])
        qv  = vec.transform([uq['keywords']])
        sims= cosine_similarity(qv, M).flatten()
        df['ir_score'] = sims
        df = df[sims>0].sort_values('ir_score', ascending=False)
    else:
        df['ir_score'] = 0.0

    # ML re-ranking
    try:
        top_n = int(uq.get('top_n', 10))
    except ValueError:
        top_n = 10

    # prepare features
    y_true = df["ratings_place"].astype(float)
    Xt  = tfidf_global.transform(df["combined_text"])
    Xci = tfidf_city.transform(df["city"])
    Xp  = tfidf_place.transform(df["place"])
    dur_c = df["ideal_duration"].apply(parse_duration).fillna(dur.mean())
    Xn = csr_matrix(scaler.transform(dur_c.values.reshape(-1,1)))
    Xc_new = hstack([Xt, Xci, Xp, Xn]).tocsr()

    df['ml_score'] = reg.predict(Xc_new)

    return df.sort_values('ml_score', ascending=False).head(top_n).reset_index(drop=True)
