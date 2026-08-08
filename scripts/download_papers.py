import json
import os
import re
from urllib.request import urlretrieve

import arxiv

LIBRARY_DIR = "./csc_library"
TARGETS = {
    "Information Retrieval": ("cs.IR", "dense retrieval OR semantic search OR RAG"),
    "Machine Learning": ("cs.LG", "transformer architecture OR quantization OR LLM inference"),
    "Distributed Computing": ("cs.DC", "raft consensus OR distributed systems OR fault tolerance"),
    "Information Theory": ("cs.IT", "shannon entropy OR channel capacity OR error correcting codes"),
    "Databases": ("cs.DB", "B-Tree optimization OR LSM-tree storage OR query compilation"),
}


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:100]


def main() -> None:
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    client = arxiv.Client()
    manifest = []

    for domain, (category, keywords) in TARGETS.items():
        print(f"\n--- Harvesting: {domain} ({category}) ---")
        search = arxiv.Search(
            query=f"cat:{category} AND ({keywords})",
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in client.results(search):
            filename = f"{sanitize_filename(result.title)}.pdf"
            filepath = os.path.join(LIBRARY_DIR, filename)
            try:
                urlretrieve(result.pdf_url, filepath)
            except Exception as e:
                print(f"Error downloading {result.title[:20]}: {e}")
                continue
            manifest.append(
                {
                    "id": result.entry_id.split("/abs/")[-1],
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published_year": result.published.year,
                    "primary_category": result.primary_category,
                    "all_categories": result.categories,
                    "local_path": filepath,
                    "url": result.pdf_url,
                }
            )

    manifest_path = os.path.join(LIBRARY_DIR, "library_metadata.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"\nDownloaded {len(manifest)} papers to {LIBRARY_DIR}")


if __name__ == "__main__":
    main()
