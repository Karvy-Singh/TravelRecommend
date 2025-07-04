import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
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
            # wrap around end of year
            return target >= start or target <= end
    else:
        return t in range_str.lower()


# 1. Read full data
df = pd.read_json(DATA_FILE, lines=True)

# 2. Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# 3. Prepare combined text
for d in (train_df, test_df):
    d["combined_text"] = (
        d["place_desc"].fillna("") + " " + d["city_desc"].fillna("")
    ).str.strip()

# 4. Text vectorizers (fit on train only)
tfidf_global = TfidfVectorizer(stop_words="english", max_features=15_000)
tfidf_global.fit(train_df["combined_text"])

tfidf_city  = TfidfVectorizer(stop_words="english", max_features=2_000)
tfidf_city.fit(train_df["city"].fillna(""))

tfidf_place = TfidfVectorizer(stop_words="english", max_features=2_000)
tfidf_place.fit(train_df["place"].fillna(""))

# 5. Transform train/test
X_text_train = tfidf_global.transform(train_df["combined_text"])
X_text_test  = tfidf_global.transform(test_df["combined_text"])

X_city_train = tfidf_city.transform(train_df["city"].fillna(""))
X_city_test  = tfidf_city.transform(test_df["city"].fillna(""))

X_place_train = tfidf_place.transform(train_df["place"].fillna(""))
X_place_test  = tfidf_place.transform(test_df["place"].fillna(""))

# 6. Numeric feature: ideal duration
dur_train = train_df["ideal_duration"].apply(parse_duration)
dur_test  = test_df["ideal_duration"].apply(parse_duration)

# fill any NaNs in train with train‐mean, same for test
mean_dur = dur_train.mean()
dur_train = dur_train.fillna(mean_dur)
dur_test  = dur_test.fillna(mean_dur)

scaler = StandardScaler()
X_num_train = csr_matrix(scaler.fit_transform(dur_train.values.reshape(-1, 1)))
X_num_test  = csr_matrix(scaler.transform(dur_test.values.reshape(-1, 1)))

# 7. Combine all features
X_train = hstack([X_text_train, X_city_train, X_place_train, X_num_train]).tocsr()
X_test  = hstack([X_text_test,  X_city_test,  X_place_test,  X_num_test ]).tocsr()

y_train = train_df["ratings_place"].astype(float).values
y_test  = test_df["ratings_place"].astype(float).values

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# 8. Train regressor
reg = XGBRegressor(
    tree_method="hist",
    max_bin=128,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.3,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=4
)

reg.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# 9. Evaluate on test set
y_pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.3f}")

y_train_pred = reg.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Train RMSE: {train_rmse:.3f}")

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
                        .apply(lambda s: (parse_duration(s) is not None)
                                       and (parse_duration(s) <= max_d))]
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

    # prepare features using the global/train‐fitted transformers
    Xt  = tfidf_global.transform(df["combined_text"])
    Xci = tfidf_city.transform(df["city"].fillna(""))
    Xp  = tfidf_place.transform(df["place"].fillna(""))
    dur_c = df["ideal_duration"].apply(parse_duration).fillna(mean_dur)
    Xn = csr_matrix(scaler.transform(dur_c.values.reshape(-1,1)))
    Xc_new = hstack([Xt, Xci, Xp, Xn]).tocsr()

    df['ml_score'] = reg.predict(Xc_new)

    return df.head(top_n).reset_index(drop=True)

