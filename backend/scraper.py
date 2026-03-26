import json
from playwright.sync_api import sync_playwright

def scrape_entertainment_news(page):
    #for entertainment news section
    try:
        print("Navigating to news page...")
        page.goto("https://ekantipur.com/news", wait_until="networkidle")
    except Exception as e:
        print(f"Error navigating to news page: {e}")
        return []

    
    # Wait for article cards to be present
    page.wait_for_selector("div.category-inner-wrapper")

    # Scroll down to trigger lazy image loading
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        page.wait_for_timeout(2000)

        # Grab all article cards
        cards = page.query_selector_all("div.category-inner-wrapper")
        print(f"Found {len(cards)} article cards")
    except Exception as e:
        print(f"Error during scrolling or grabbing article cards: {e}")
        cards = []

    article_links = []
    category_el = page.query_selector("div.category-name p a")
    category = category_el.text_content().strip()
    for card in cards[:5]: 
        try:
            #grab title inside category-description h2->a tag
            url_el =  card.query_selector("div.category-description h2 a")
            url = url_el.get_attribute("href") if url_el else None
            print(url)
            title_el = card.query_selector("div.category-description h2 a")
            title = title_el.text_content().strip() if title_el else None
            print(title)
            author_el = card.query_selector("div.author-name p a")
            author = author_el.text_content().strip() if author_el else None
            print(author)
            if url:
                article_links.append({
                    "url":url,
                    "title":title,
                    "author":author
                })
        except Exception as e:
            print(f"Error: {e}")

    articles=[]
    for item in article_links:
        try:
            page.goto(item['url'], wait_until = "networkidle")
            page.wait_for_selector(".news-inner-wrapper")
            paragraphs = page.query_selector_all(".news-inner-wrapper p")
            article = " ".join([
                (p.text_content()).strip()
                for p in paragraphs
            ])
            print(article[:200])
            articles.append({
                "title":item['title'],
                "url":item['url'],
                "author":item['author'],
                "article": article,
                "category": category
            })
        except Exception as e:
            print(f"Erro: {e}")
        print(articles)

    return articles




def main():
    with sync_playwright() as p:
       
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        entertainment_news = scrape_entertainment_news(page)
        print(f"\nExtracted {len(entertainment_news)} news articles")


        browser.close()

        output = {
            "news": entertainment_news,
        }

        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("\nDone! Data saved to output.json")


if __name__ == "__main__":
    main()
