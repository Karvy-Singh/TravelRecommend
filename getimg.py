import requests

def is_valid_image_url(url: str) -> bool:
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        return response.status_code == 200 and content_type.startswith("image/")
    except requests.RequestException:
        return False

def fetch_google_images(query: str, limit: int = 5):
    API_KEY = "AIzaSyAx0g4SAJmUGgUoU2soXAY0YZmr2_Ia3JE"
    CX = "b332fc4cf2f664bac"
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": CX,
        "key": API_KEY,
        "searchType": "image",
        "num": limit
    }

    response = requests.get(search_url, params=params)
    response.raise_for_status()
    data = response.json()

    raw_image_urls = [item["link"] for item in data.get("items", [])]
    valid_image_urls = [url for url in raw_image_urls if is_valid_image_url(url)]

    return valid_image_urls




