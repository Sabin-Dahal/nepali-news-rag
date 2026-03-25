import json
import re
import unicodedata

def clean_text(text):
    if not text:
        return ""
    # Remove zero-width / invisible chars
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    # Replace non-breaking spaces with normal space
    text = text.replace('\u00A0', ' ')
    # Normalize Unicode (important for Nepali)
    text = unicodedata.normalize('NFKC', text)
    # Strip leading/trailing spaces
    return text.strip()

# Load old JSON
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
articles = data.get("news", [])
# Clean each article
for article in articles:
    for key in ['title', 'author', 'category', 'article']:
        article[key] = clean_text(article.get(key, ""))


# Save cleaned JSON
with open('clean_output.json', 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print("Cleaned JSON saved as articles_clean.json")