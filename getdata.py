from pathlib import Path
import pandas as pd
import unicodedata
import re

DATA_DIR = Path("./data/")            
OUT_FILE = "famous_indian_tourist_places_3000.jsonl"
TARGET_ROWS = 3_000

def to_snake(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = re.sub(r"[^\w\s]", " ", name)          # drop punctuation
    name = re.sub(r"\s+", "_", name)              # whitespace → underscore
    return name.lower().strip("_")

# load
place_df = pd.read_csv(DATA_DIR / "Places.csv")
city_df  = pd.read_csv(DATA_DIR / "City.csv")

# harmonise column names
place_df.columns = [to_snake(c) for c in place_df.columns]
city_df.columns  = [to_snake(c) for c in city_df.columns]

# determine join key
if "city_id" in place_df.columns and "city_id" in city_df.columns:
    join_key = "city_id"
else:
    common = set(place_df.columns) & set(city_df.columns)
    if not common:
        raise ValueError(
            "No shared column found between Places.csv and City.csv. "
            "Please inspect the files and adjust `join_key` manually."
        )
    join_key = sorted(common)[0]
    print(f"Joining on column: {join_key!r}")

# merge
df = place_df.merge(city_df, on=join_key, how="left", suffixes=("_place", "_city"))

# drop columns that are entirely NaN
df = df.dropna(axis=1, how="all")

# strip whitespace in object fields
for col, dtype in df.dtypes.items():
    if dtype == "object":
        df[col] = df[col].astype(str).str.strip()

# convert data-types where sensible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# fill all remaining NaNs with 0
df = df.fillna(0)
# drop any exact duplicate rows
df = df.drop_duplicates(ignore_index=True)
# if more than TARGET_ROWS, keep only the first TARGET_ROWS rows
df = df.head(TARGET_ROWS)

df.to_json(OUT_FILE, orient="records", lines=True, force_ascii=False)
print(f"✔  Wrote {len(df)} rows to {OUT_FILE}")

