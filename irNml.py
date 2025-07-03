import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from xgboost import XGBRegressor

DATA_FILE = "famous_indian_tourist_places_3000.jsonl"

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
X_text_full = tfidf_global.fit_transform(df["combined_text"])

# 3. Numeric feature: ideal duration
dur = df["ideal_duration"].apply(parse_duration).fillna(df["ideal_duration"].apply(parse_duration).mean())
dur = dur.values.reshape(-1, 1)
scaler = StandardScaler()
X_num_full = csr_matrix(scaler.fit_transform(dur))

# 4. Combine
X_full = hstack([X_text_full, X_num_full]).tocsr()
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

def filter_and_search(df):
    df = df.copy()
    df['combined_text'] = (
        df['place_desc'].fillna('') + ' ' + df['city_desc'].fillna('')
    ).str.strip()

    consent= input("Want to apply hard filters? [y/n]")

    if consent== 'Y'.lower():
        uq = {
            "city"       : input("City name (or blank): "),
            "place"      : input("Place name (or blank): "),
            "rating_min" : input("Min rating (or blank): "),
            "duration_max":input("Max stay days (or blank): "),
            "month"      : input("Month to visit (or blank): "),
            "keywords"   : input("Keywords (or blank): "),
            "top_n"      : input("How many results?: ")
            }

        # … your hard filters on city/place/rating_min/duration_max/month …
        if uq['city']:
            df = df[df['city'].str.contains(uq['city'], case=False, na=False)]

        if uq['place']:
            df = df[df['place'].str.contains(uq['place'], case=False, na=False)]

        if uq['rating_min']:
            try:
                min_rating = float(uq['rating_min'])
                df = df[df['ratings_place'] >= min_rating]
            except ValueError:
                pass

        if uq['duration_max']:
            try:
                max_days = int(uq['duration_max'])
                # keep only rows whose parsed max_duration ≤ max_days
                df = df[df['ideal_duration']
                        .apply(lambda s: (parse_duration(s)[1] is not None)
                                       and (parse_duration(s)[1] <= max_days)
                              )]
            except ValueError:
                pass

        if uq['month']:
            df = df[df['best_time_to_visit']
                    .fillna('')
                    .apply(lambda rng: in_month_range(rng, uq['month']))]

    else:
        uq={
            "keywords"   : input("Keywords: "),
            "top_n"      : input("How many results?: ")
        }

    # IR step: cosine‐sim on keywords
    if uq['keywords'] and not df.empty:
        vec = TfidfVectorizer(stop_words='english')
        M   = vec.fit_transform(df['combined_text'])
        qv  = vec.transform([uq['keywords']])
        sims= cosine_similarity(qv, M).flatten()
        df['ir_score'] = sims
        df = df[sims>0].sort_values('ir_score', ascending=False)
    else:
        df['ir_score'] = 0.0

    try:
        top_n = int(uq['top_n'])
    except:
        top_n = 10

    return df.head(top_n).reset_index(drop=True)

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return np.sum(rels * discounts)

def ndcg_at_k(y_true, y_pred, k):
    order = np.argsort(-y_pred)
    ideal = np.argsort(-y_true)
    dcg   = dcg_at_k(y_true[order], k)
    idcg  = dcg_at_k(y_true[ideal], k)
    return dcg / idcg if idcg > 0 else 0.0

def precision_recall_at_k(y_true, y_pred, k, thresh=4.0):
    idx = np.argsort(-y_pred)[:k]
    rel = (y_true[idx] >= thresh).astype(int)
    precision = rel.mean()
    recall    = rel.sum() / max((y_true >= thresh).sum(), 1)
    return precision, recall

def average_precision_at_k(y_true, y_pred, k, thresh=4.0):
    idx = np.argsort(-y_pred)[:k]
    rel = (y_true[idx] >= thresh).astype(int)
    if rel.sum() == 0:
        return 0.0
    cum = [rel[:i+1].mean() for i in range(len(rel)) if rel[i]]
    return float(np.mean(cum))

def main():
    # load fresh copy for interactive query
    df_query = pd.read_json(DATA_FILE, lines=True)

    cands = filter_and_search(df_query)
    if cands.empty:
        print("No matches.")
        return

    # Baseline metrics (IR only)
    y_true    = cands["ratings_place"].values.astype(float)
    y_ir      = cands["ir_score"].values.astype(float)
    K         = len(cands)
    print(f"\nIR‐only:   NDCG@{K} {ndcg_at_k(y_true,y_ir,K):.4f},  "
          f"P@{K} {precision_recall_at_k(y_true,y_ir,K)[0]:.4f},  "
          f"MAP@{K} {average_precision_at_k(y_true,y_ir,K):.4f}")

    # ML re‐ranking: predict “rating” for each candidate
    # 1) text features
    Xt  = tfidf_global.transform(cands["combined_text"])
    Xci = tfidf_global.transform(cands["city"])
    Xp  = tfidf_global.transform(cands["place"])
    dur_c = cands["ideal_duration"].apply(parse_duration).fillna(dur.mean())
    Xn = csr_matrix(scaler.transform(dur_c.values.reshape(-1,1)))

    Xc_new = hstack([Xt, Xci, Xp, Xn]).tocsr()

# 2. retrain your regressor on the new, wider matrix (and true ratings y)
    reg.fit(Xc_new, cands["ratings_place"])

# 3. now predict
    cands["ml_score"] = reg.predict(Xc_new)
    # ML metrics
    y_ml = cands["ml_score"].values
    print(f"ML‐ranker: NDCG@{K} {ndcg_at_k(y_true,y_ml,K):.4f},  "
          f"P@{K} {precision_recall_at_k(y_true,y_ml,K)[0]:.4f},  "
          f"MAP@{K} {average_precision_at_k(y_true,y_ml,K):.4f}")

    # show results
    print("\nTop re-ranked:")
    for _, r in cands.sort_values("ml_score", ascending=False).iterrows():
        print(f" • {r['place']} ({r['city']}) — rating {r['ratings_place']:.1f},  score {r['ml_score']:.3f}")
        print("\n" ,f"{r['best_time_to_visit']} and {r['ideal_duration']}")

if __name__ == "__main__":
    main()

