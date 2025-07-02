import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBRegressor
from typing import Tuple

DATA_FILE   = "famous_indian_tourist_places_3000.jsonl"
TEXT_COLUMNS = ["place_desc", "city_desc"]
NUMERIC_COLUMNS = [
    "ratings_place",          # already numeric
    "ideal_duration_parsed"   # introduced below
]
RELEVANCE_THRESHOLD = 4.0     # rating ≥ 4  ⇒ relevant
TOP_K = 10                    # evaluate @10

def load_and_prepare(path: str | Path) -> Tuple[csr_matrix, np.ndarray]:
    """Load jsonl, build feature matrix X and target y (ratings)."""
    df = pd.read_json(path, lines=True)

    df["combined_text"] = (
        df["place_desc"].fillna("").astype(str).str.strip() + " " +
        df["city_desc"].fillna("").astype(str).str.strip()
    ).str.strip()

    tfidf = TfidfVectorizer(stop_words="english", max_features=15_000)
    X_text = tfidf.fit_transform(df["combined_text"])

    # ideal_duration can be strings like "2-4" or "3"; parse → mean days
    def _parse_duration(s: str | float | int) -> float:
        if isinstance(s, str):
            s = s.strip()
            if "-" in s:
                try:
                    low, high = map(float, s.split("-", 1))
                    return (low + high) / 2
                except ValueError:
                    return np.nan
            try:
                return float(s)
            except ValueError:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    df["ideal_duration_parsed"] = df.get("ideal_duration", np.nan).apply(_parse_duration)

    num_df = df[NUMERIC_COLUMNS].copy()
    # only keep columns that actually exist
    num_df = num_df[[c for c in NUMERIC_COLUMNS if c in num_df.columns]]

    scaler = StandardScaler()
    X_num = scaler.fit_transform(num_df.fillna(num_df.mean()))
    X_num = csr_matrix(X_num)               # make it sparse to hstack

    # Final feature matrix
    X = hstack([X_text, X_num]).tocsr()

    # Target: rating (float) – will be used as graded relevance
    y = df["ratings_place"].values.astype(float)

    return X, y

def dcg_at_k(rels: np.ndarray, k: int) -> float:
    """Discounted cumulative gain."""
    rels = rels[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return np.sum(rels * discounts)

def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k with graded relevance (here: ratings)."""
    idx     = np.argsort(-y_pred)                  # best first
    ideal   = np.argsort(-y_true)
    dcg     = dcg_at_k(y_true[idx],   k)
    idcg    = dcg_at_k(y_true[ideal], k)
    return dcg / idcg if idcg > 0 else 0.0

def precision_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                          k: int, thresh: float) -> Tuple[float, float]:
    """Binary relevance by threshold."""
    idx   = np.argsort(-y_pred)[:k]
    rel   = (y_true[idx] >= thresh).astype(int)
    precision = rel.mean()
    recall    = rel.sum() / max((y_true >= thresh).sum(), 1)
    return precision, recall

def average_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                           k: int, thresh: float) -> float:
    idx   = np.argsort(-y_pred)[:k]
    rel   = (y_true[idx] >= thresh).astype(int)
    if rel.sum() == 0:
        return 0.0
    cum_precisions = [
        rel[:i + 1].mean() for i in range(len(rel)) if rel[i]
    ]
    return np.mean(cum_precisions)

def main() -> None:
    print("Loading and vectorising data …")
    X, y = load_and_prepare(DATA_FILE)

    print("Splitting train / test …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost …")
    model = XGBRegressor(
    tree_method="hist",       # 1️⃣ histogram grower
    max_bin=128,              # 2️⃣ coarser hist bins
    n_estimators=200,         # 3️⃣ fewer trees
    learning_rate=0.15,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    n_jobs=4,                 # 4️⃣ leave spare cores
    random_state=42
)
    model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
    print("Predicting on test set …")
    y_pred = model.predict(X_test)

    ndcg  = ndcg_at_k(y_test, y_pred, TOP_K)
    prec, rec = precision_recall_at_k(y_test, y_pred, TOP_K, RELEVANCE_THRESHOLD)
    map_k = average_precision_at_k(y_test, y_pred, TOP_K, RELEVANCE_THRESHOLD)

    print(f"\nRanking metrics @{TOP_K}")
    print(f"  NDCG  : {ndcg:0.4f}")
    print(f"  MAP   : {map_k:0.4f}")
    print(f"  Precision: {prec:0.4f}")
    print(f"  Recall   : {rec:0.4f}")

if __name__ == "__main__":
    main()

