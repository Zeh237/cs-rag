import os
import json
import re
import arxiv
from urllib.request import urlretrieve

# 1. Set up directories
LIBRARY_DIR = "./csc_library"
os.makedirs(LIBRARY_DIR, exist_ok=True)

# Initialize the modern arXiv API Client
client = arxiv.Client()

# 2. Map domains to categories and distinct search keywords
TARGET_DOMAINS = {
    "Information Retrieval": {
        "category": "cs.IR",
        "keywords": "dense retrieval OR semantic search OR RAG"
    },
    "Machine Learning": {
        "category": "cs.LG",
        "keywords": "transformer architecture OR quantization OR LLM inference"
    },
    "Distributed Computing": {
        "category": "cs.DC",
        "keywords": "raft consensus OR distributed systems OR fault tolerance"
    },
    "Information Theory": {
        "category": "cs.IT",
        "keywords": "shannon entropy OR channel capacity OR error correcting codes"
    },
    "Databases": {
        "category": "cs.DB",
        "keywords": "B-Tree optimization OR LSM-tree storage OR query compilation"
    }
}

metadata_manifest = []


# Clean filenames to avoid filesystem errors
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)[:100]


# 3. Query and download papers for each domain
for domain, config in TARGET_DOMAINS.items():
    print(f"\n--- Harvesting: {domain} ({config['category']}) ---")

    # Construct a targeted boolean query
    query = f"cat:{config['category']} AND ({config['keywords']})"

    search = arxiv.Search(
        query=query,
        max_results=5,  # Adjust this to scale your dataset
        sort_by=arxiv.SortCriterion.Relevance
    )

    # Execute search using the client instance
    for result in client.results(search):
        clean_title = sanitize_filename(result.title)
        filename = f"{clean_title}.pdf"
        filepath = os.path.join(LIBRARY_DIR, filename)

        # Download the PDF using standard urllib.request
        print(f"-> Downloading: {result.title[:60]}...")
        try:
            urlretrieve(result.pdf_url, filepath)
        except Exception as e:
            print(f"   [Error downloading {result.title[:20]}]: {e}")
            continue

        # Structure the exact metadata needed for Elasticsearch indexing
        metadata_manifest.append({
            "id": result.entry_id.split("/abs/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published_year": result.published.year,
            "primary_category": result.primary_category,
            "all_categories": result.categories,
            "local_path": filepath,
            "url": result.pdf_url
        })

# 4. Save metadata package
manifest_path = os.path.join(LIBRARY_DIR, "library_metadata.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(metadata_manifest, f, indent=4, ensure_ascii=False)

print(f"\nSuccess! Downloaded {len(metadata_manifest)} papers.")
print(f"Metadata manifest written to: {manifest_path}")