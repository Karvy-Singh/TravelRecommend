import requests

def fetch_wiki_images(page_title: str, limit: int = 5):
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    # 1) get all image titles on that page
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "format": "json",
        "imlimit": limit
    }
    r = S.get(URL, params=params).json()
    pages = next(iter(r["query"]["pages"].values()))
    titles = [img["title"] for img in pages.get("images", [])]

    # 2) fetch their direct URLs
    params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    r = S.get(URL, params=params).json()
    out = []
    for p in r["query"]["pages"].values():
        info = p.get("imageinfo", [])
        if info:
            out.append(info[0]["url"])
    return out
 

