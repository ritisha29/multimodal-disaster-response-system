"""
Disaster Type Classification & Location Extraction Module.

Second-stage post-processing for informative disaster tweets:
1. Keyword-based disaster type categorization (Flood, Wildfire, Earthquake/Landslide, Cyclone/Hurricane, Other).
2. Hybrid Location Extraction:
   - Indian Gazetteer (cities, nicknames, states, UTs) for fast, robust recognition of informal place names (e.g. "Vizag" -> "Visakhapatnam", "Andhra Pradesh", "Bangalore", etc.).
   - Named Entity Recognition (NER) via spaCy (en_core_web_sm) to extract GPE/LOC location mentions.
   - Merged deduplicated location detection with clear fallback to "Location not specified".
"""

import re
from typing import Dict, List, Any

# Load spaCy NLP model lazily with fallback
_NLP_MODEL = None


def get_nlp_model():
    """Lazily load spaCy model en_core_web_sm with graceful fallback."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        try:
            import spacy
            _NLP_MODEL = spacy.load("en_core_web_sm")
        except Exception:
            try:
                import spacy
                spacy.cli.download("en_core_web_sm")
                _NLP_MODEL = spacy.load("en_core_web_sm")
            except Exception:
                _NLP_MODEL = False
    return _NLP_MODEL


DISASTER_KEYWORDS = {
    "Flood": {
        "keywords": [
            "flood", "floods", "flooding", "flooded", "flash flood", "flash floods",
            "water", "waters", "river overflow", "submerged", "inundated", "inundation",
            "deluge", "drowning", "dam break", "waterlogging", "cloudburst", "cloudbursts"
        ],
        "icon": "🌊"
    },
    "Wildfire": {
        "keywords": [
            "fire", "fires", "wildfire", "wildfires", "flames", "smoke", "blaze", "blazes",
            "burn", "burning", "burnt", "bushfire", "bushfires", "firefighters", "forest fire", "forest fires"
        ],
        "icon": "🔥"
    },
    "Earthquake / Landslide": {
        "keywords": [
            "earthquake", "earthquakes", "quake", "quakes", "tremor", "tremors", "aftershock",
            "aftershocks", "seismic", "shaking", "epicenter", "tsunami", "tsunamis",
            "landslide", "landslides", "mudslide", "mudslides", "rockslide"
        ],
        "icon": "🌋"
    },
    "Cyclone / Hurricane": {
        "keywords": [
            "cyclone", "cyclones", "hurricane", "hurricanes", "typhoon", "typhoons", "storm",
            "storms", "wind", "winds", "gale", "gales", "tropical storm", "storm surge",
            "monsoon", "monsoons", "heavy rain", "heavy rains", "downpour"
        ],
        "icon": "🌀"
    }
}

# Indian Gazetteer: Maps informal nicknames, city variants, states, and UTs to canonical display names
INDIAN_GAZETTEER = {
    # Major City Nicknames & Variants
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "waltair": "Visakhapatnam",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "madras": "Chennai",
    "chennai": "Chennai",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "gauhati": "Guwahati",
    "guwahati": "Guwahati",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "baroda": "Vadodara",
    "vadodara": "Vadodara",
    "benaras": "Varanasi",
    "varanasi": "Varanasi",
    "kashi": "Varanasi",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "calicut": "Kozhikode",
    "kozhikode": "Kozhikode",
    "trichy": "Tiruchirappalli",
    "tiruchirappalli": "Tiruchirappalli",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "secunderabad": "Secunderabad",
    "hyderabad": "Hyderabad",
    "ahmedabad": "Ahmedabad",
    "amdavad": "Ahmedabad",
    "pune": "Pune",
    "poona": "Pune",
    "nagpur": "Nagpur",
    "surat": "Surat",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "kanpur": "Kanpur",
    "patna": "Patna",
    "bhubaneswar": "Bhubaneswar",
    "bhubaneshwar": "Bhubaneswar",
    "cuttack": "Cuttack",
    "ranchi": "Ranchi",
    "jamshedpur": "Jamshedpur",
    "shillong": "Shillong",
    "imphal": "Imphal",
    "agartala": "Agartala",
    "aizawl": "Aizawl",
    "kohima": "Kohima",
    "gangtok": "Gangtok",
    "itanagar": "Itanagar",
    "dehradun": "Dehradun",
    "shimla": "Shimla",
    "simla": "Shimla",
    "srinagar": "Srinagar",
    "jammu": "Jammu",
    "leh": "Leh",
    "vijayawada": "Vijayawada",
    "bezawada": "Vijayawada",
    "guntur": "Guntur",
    "tirupati": "Tirupati",
    "kakinada": "Kakinada",
    "nellore": "Nellore",
    "kurnool": "Kurnool",
    "anantapur": "Anantapur",
    "rajahmundry": "Rajahmundry",
    "rajamahendravaram": "Rajahmundry",
    "warangal": "Warangal",
    "nizamabad": "Nizamabad",
    "karimnagar": "Karimnagar",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "mangalore": "Mangaluru",
    "mangaluru": "Mangaluru",
    "hubli": "Hubballi",
    "hubballi": "Hubballi",
    "belgaum": "Belagavi",
    "belagavi": "Belagavi",
    "coimbatore": "Coimbatore",
    "kovai": "Coimbatore",
    "madurai": "Madurai",
    "salem": "Salem",

    # Indian States & Union Territories
    "andhra pradesh": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "telangana": "Telangana",
    "karnataka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "kerala": "Kerala",
    "maharashtra": "Maharashtra",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "uttar pradesh": "Uttar Pradesh",
    "uttarpradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "bihar": "Bihar",
    "west bengal": "West Bengal",
    "westbengal": "West Bengal",
    "bengal": "West Bengal",
    "assam": "Assam",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "jharkhand": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "himachal pradesh": "Himachal Pradesh",
    "himachal": "Himachal Pradesh",
    "jammu and kashmir": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "delhi": "Delhi",
    "ncr": "Delhi",
    "new delhi": "Delhi",
    "goa": "Goa",
    "tripura": "Tripura",
    "meghalaya": "Meghalaya",
    "manipur": "Manipur",
    "nagaland": "Nagaland",
    "mizoram": "Mizoram",
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",
    "sikkim": "Sikkim",
    "chandigarh": "Chandigarh",
    "andaman and nicobar": "Andaman and Nicobar",
    "andaman": "Andaman and Nicobar"
}


def classify_disaster_type(tweet_text: str) -> Dict[str, Any]:
    """
    Classifies the disaster type based on flexible, case-insensitive keyword matching.

    Args:
        tweet_text (str): Input tweet text.

    Returns:
        dict: Contains 'disaster_type', 'icon', and 'matched_keywords'.
    """
    text_lower = tweet_text.lower()
    matched_results = {}

    for dtype, info in DISASTER_KEYWORDS.items():
        matched = []
        for kw in info["keywords"]:
            # Check exact word boundary match or word boundary with optional 's'/'ing'/'ed'
            pattern = r'\b' + re.escape(kw) + r'(?:s|ing|ed)?\b'
            if re.search(pattern, text_lower):
                matched.append(kw)
        if matched:
            matched_results[dtype] = matched

    if not matched_results:
        return {
            "disaster_type": "Other / General Emergency",
            "icon": "🚨",
            "matched_keywords": []
        }

    # Pick disaster type with the most matched keywords
    best_match = max(matched_results.keys(), key=lambda k: len(matched_results[k]))
    return {
        "disaster_type": best_match,
        "icon": DISASTER_KEYWORDS[best_match]["icon"],
        "matched_keywords": matched_results[best_match]
    }


def extract_locations(tweet_text: str) -> str:
    """
    Extracts location mentions using a hybrid approach:
    1. Indian Gazetteer lookup for informal names, city nicknames, states, and UTs.
    2. Named Entity Recognition (NER) via spaCy (GPE and LOC entities).

    Args:
        tweet_text (str): Input tweet text.

    Returns:
        str: Comma-separated list of extracted locations or "Location not specified".
    """
    text_lower = tweet_text.lower()
    found_locations = []

    # 1. Gazetteer Lookup
    # Sort keys by length descending to match multi-word entries like "andhra pradesh" before "andhra"
    sorted_gaz_keys = sorted(INDIAN_GAZETTEER.keys(), key=len, reverse=True)
    matched_canonical = set()

    for key in sorted_gaz_keys:
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, text_lower):
            canonical = INDIAN_GAZETTEER[key]
            if canonical not in matched_canonical:
                matched_canonical.add(canonical)
                found_locations.append(canonical)

    # 2. spaCy NER Lookup
    nlp = get_nlp_model()
    if nlp:
        try:
            doc = nlp(tweet_text)
            for ent in doc.ents:
                if ent.label_ in ("GPE", "LOC"):
                    loc_clean = ent.text.strip()
                    if loc_clean and loc_clean not in found_locations and loc_clean.lower() not in INDIAN_GAZETTEER:
                        found_locations.append(loc_clean)
        except Exception:
            pass

    if found_locations:
        return ", ".join(found_locations)

    return "Location not specified"


def analyze_disaster_info(tweet_text: str) -> Dict[str, Any]:
    """
    Combined second-stage analysis function returning disaster type and extracted locations.

    Args:
        tweet_text (str): Input tweet text.

    Returns:
        dict: High-level analysis results.
    """
    dtype_info = classify_disaster_type(tweet_text)
    location_str = extract_locations(tweet_text)

    return {
        "disaster_type": dtype_info["disaster_type"],
        "icon": dtype_info["icon"],
        "matched_keywords": dtype_info["matched_keywords"],
        "location": location_str
    }


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def get_clip_components():
    """Lazily loads CLIP model and processor for zero-shot image classification."""
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception:
            _CLIP_MODEL = False
            _CLIP_PROCESSOR = False
    return _CLIP_MODEL, _CLIP_PROCESSOR


def classify_image_disaster_type(image_obj=None, filename: str = "") -> str:
    """
    Classifies an uploaded image into one of:
    - Flood
    - Wildfire
    - Earthquake / Landslide
    - Cyclone / Hurricane
    - Unclear

    Uses CLIP zero-shot classification if available, with filename fallback.
    """
    # 1. Filename heuristic fallback if image_obj is None or invalid
    fn_lower = filename.lower() if filename else ""
    if "flood" in fn_lower or "water" in fn_lower or "inundat" in fn_lower:
        return "Flood"
    elif "fire" in fn_lower or "flame" in fn_lower or "smoke" in fn_lower or "burn" in fn_lower:
        return "Wildfire"
    elif "quake" in fn_lower or "slide" in fn_lower or "landslide" in fn_lower:
        return "Earthquake / Landslide"
    elif "storm" in fn_lower or "cyclone" in fn_lower or "hurricane" in fn_lower:
        return "Cyclone / Hurricane"

    if image_obj is None:
        return "Unclear"

    # 2. CLIP zero-shot classification
    model, processor = get_clip_components()
    if not model or not processor:
        return "Unclear"

    try:
        candidate_labels = [
            "a photo of a flood with submerged houses and water overflow",
            "a photo of a wildfire with bright flames, fire, and smoke",
            "a photo of earthquake damage, collapsed buildings, or a landslide",
            "a photo of a cyclone, severe hurricane storm, or heavy rain",
            "a photo of a person, indoor scene, document, logo, or generic photo"
        ]
        label_map = {
            candidate_labels[0]: "Flood",
            candidate_labels[1]: "Wildfire",
            candidate_labels[2]: "Earthquake / Landslide",
            candidate_labels[3]: "Cyclone / Hurricane",
            candidate_labels[4]: "Unclear"
        }

        import torch
        inputs = processor(text=candidate_labels, images=image_obj, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape (1, num_labels)
            probs = logits_per_image.softmax(dim=1)[0]   # shape (num_labels,)

        best_idx = int(torch.argmax(probs).item())
        best_prob = float(probs[best_idx].item())
        best_label = candidate_labels[best_idx]

        mapped_cat = label_map[best_label]
        if mapped_cat != "Unclear" and best_prob < 0.28:
            return "Unclear"

        return mapped_cat
    except Exception:
        return "Unclear"


if __name__ == "__main__":
    test_input = "floods in vizag / floods in andhra pradesh"
    print(f"Testing input: '{test_input}'")
    res = analyze_disaster_info(test_input)
    print("Disaster Type:", res["disaster_type"])
    print("Matched Keywords:", res["matched_keywords"])
    print("Extracted Location:", res["location"])

