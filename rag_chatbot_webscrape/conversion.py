from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import html2text
import os
from urllib.parse import urljoin, urlparse
import time

# base url and starting point
base_url = "https://kubernetes.io/docs/_print/"
docs_base = "https://kubernetes.io/docs"

# directory to save markdown files
output_dir = "kubernetes_docs"
os.makedirs(output_dir, exist_ok=True)

# initialize html2text converter
h2t = html2text.HTML2Text()
h2t.body_width = 0 # disable line wrapping
h2t.ignore_links = False # keep links
h2t.ignore_images = True # ignore images

# set to keep track of visited urls
visited_urls = set()

def sanitize_url(url):
    """convert url to a valid filename"""
    parsed = urlparse(url)
    path = parsed.path.strip('/').replace('/', '-') or "index"
    return f"{path}.md"

def save_markdown(url, content):
    """save content to markdown file"""
    filename = os.path.join(output_dir, sanitize_url(url))
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"saved markdown file: {filename}")
    except IOError as e:
        print(f"Error writing to file: {filename}: {e}")

def fetch_and_convert(url):
    """fetch a url, convert to markdown, and return links found"""
    if url in visited_urls:
        return []
    visited_urls.add(url)
    print(f"fetching {url}")
    try:
        loader = WebBaseLoader(url, headers={"User-Agent": "Mozilla/5.0"})
        documents = loader.load()
        if not documents:
            print(f"No content loaded from {url}")
            return []
        content = documents[0].page_content
    except Exception as e:
        print(f"Error fetching: {e}")
        return []
    # convert html to markdown
    markdown_content = h2t.handle(content)
    save_markdown(url, markdown_content)
    # extract links from content
    soup = BeautifulSoup(content, "html.parser")
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_url = urljoin(url, href)
        if absolute_url.startswith(docs_base) and absolute_url not in visited_urls:
            links.append(absolute_url)
    return links

def crawl_docs(start_url, max_pages=None, delay=1):
    """crawl documentation pages starting from start_url"""
    urls_to_visit = [start_url]
    pages_crawled = 0

    while urls_to_visit :
        if max_pages is not None and pages_crawled >= max_pages:
            print(f"Reached max_pages limit ({max_pages}). Stopping.")
            break
        
        current_url = urls_to_visit.pop(0)
        new_links = fetch_and_convert(current_url)
        urls_to_visit.extend(new_links)
        pages_crawled += 1
        print(f"Pages crawled: {pages_crawled}, Links to visit: {len(urls_to_visit)}")
        time.sleep(delay) 
    print(f"crawling complete. Total pages crawled: {pages_crawled}")

# start crawling
crawl_docs(base_url)


