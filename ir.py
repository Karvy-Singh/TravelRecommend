import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Map month names to numbers for easy comparison
MONTH_MAP = {
    m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June",
         "July","August","September","October","November","December"],
        start=1
    )
}

def parse_duration(s):
    """
    Parse strings like "2-4" into (min_days, max_days).
    Returns (None, None) if not parseable.
    """
    if isinstance(s, str) and '-' in s:
        parts = s.split('-', 1)
        try:
            low = float(parts[0])
            high = float(parts[1])
            return low, high
        except ValueError:
            return None, None
    try:
        v = float(s)
        return v, v
    except:
        return None, None

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

def get_user_input():
    print("Enter your travel preferences (leave blank to skip a filter):\n")
    return {
        "city":       input("City name: ").strip(),
        "place":      input("Place name: ").strip(),
        "rating_min": input("Minimum place rating (e.g. 4): ").strip(),
        "duration_max": input("Maximum number of days stay: ").strip(),
        "month":      input("Time to visit (month): ").strip(),
        "keywords":   input("Search keywords (e.g. 'healing backwater resort'): ").strip(),
        "top_n":      input("How many top results would you like?: ").strip(),
    }

def filter_and_search(df, uq):
    df = df.copy()

    # Combine city & place descriptions
    df['combined_text'] = (
        df['place_desc'].fillna('').astype(str).str.strip()
        + ' '
        + df['city_desc'].fillna('').astype(str).str.strip()
    ).str.strip()

    # Hard filters
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

    #  TF-IDF similarity (if keywords provided)
    if uq['keywords']:
        if df.empty:
            print("No documents left after filtering—skipping keyword similarity step.")
            df['similarity'] = None
        else:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
            query_vec   = vectorizer.transform([uq['keywords']])
            sims        = cosine_similarity(query_vec, tfidf_matrix).flatten()

            df['similarity'] = sims
            # only keep non-zero similarities, sorted descending
            df = df[df['similarity'] > 0].sort_values('similarity', ascending=False)
    else:
        df['similarity'] = None

    # 4) Top-N selection
    try:
        top_n = int(uq['top_n']) if uq['top_n'] else 10
    except ValueError:
        top_n = 10

    return df.head(top_n)

def main():
    # Load your data
    df = pd.read_json("./famous_indian_tourist_places_3000.jsonl",
                      lines=True)

    uq = get_user_input()
    results = filter_and_search(df, uq)

    if results.empty:
        print("\nNo matching travel recommendations found.")
    else:
        print("\nTop Matches:\n")
        for _, row in results.iterrows():
            print(f"{row['place']} in {row['city']} (Rating: {row['ratings_place']})")
            print(f" Description: {row['place_desc']}...")
            if row['similarity'] is not None:
                print(f"Similarity Score: {row['similarity']:.4f}")
            print("-" * 60)

if __name__ == "__main__":
    main()

