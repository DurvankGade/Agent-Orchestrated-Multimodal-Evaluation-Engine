import re

def normalize(text: str) -> str:
    """
    Standardizes text by:
    1. Lowercasing
    2. Removing all punctuation/special characters
    3. Collapsing multiple spaces into one
    4. Stripping leading/trailing whitespace
    """
    if not text:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove non-alphanumeric (except spaces)
    text = re.sub(r"[^\w\s]", "", text)
    
    # 3. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
