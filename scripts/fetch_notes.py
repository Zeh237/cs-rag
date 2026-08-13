"""Fetch the PDF-backed open undergraduate computer science notes into ./csc_notes.

Each source is a PDF. The script downloads the PDF (stored locally under
csc_notes/pdfs/), normalizes its text to a Markdown file under csc_notes/, and
registers both in csc_notes/library_metadata.json so it can be indexed by
scripts/index_library.py (which requires a local PDF per note).

Only PDF sources are fetched: the RAG app links every result to a locally
stored PDF, so marks/etc. without a local PDF are intentionally excluded.

Sources:
  - Open Data Structures (Python edition)  (Data Structures, PDF)
  - Algorithms (Jeff Erickson)             (Algorithms, PDF)
  - Models of Computation (Erickson)       (Theory of Computation, PDF)
  - A Course in Machine Learning           (Intro ML, PDF)
  - Think OS                                (Operating Systems, PDF)
  - An Introduction to Computer Networks    (Networking, PDF)
  - Basics of Compiler Design               (Compilers, PDF)
  - Discrete Mathematics: An Open Introduction (Discrete Math, PDF)
"""

import json
import re
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.page_cleaner import clean_page_texts
NOTES_DIR = PROJECT_ROOT / "csc_notes"
MANIFEST_PATH = NOTES_DIR / "library_metadata.json"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

SOURCES = [
    {
        "category": "Data Structures",
        "title": "Open Data Structures (Python edition)",
        "authors": "Pat Morin",
        "license": "CC BY 2.5 CA",
        "kind": "pdf",
        "pages": [{"name": "open_data_structures", "url": "https://opendatastructures.org/ods-python.pdf"}],
        "file_prefix": "ds_open_data_structures",
    },
    {
        "category": "Algorithms",
        "title": "Algorithms",
        "authors": "Jeff Erickson",
        "license": "CC BY 4.0",
        "kind": "pdf",
        "pages": [
            {"name": n, "url": f"https://jeffe.cs.illinois.edu/teaching/algorithms/book/{n}.pdf"}
            for n in [
                "00-intro", "01-recursion", "02-backtracking", "03-dynprog", "04-greedy",
                "05-graphs", "06-dfs", "07-mst", "08-sssp", "09-apsp",
                "10-maxflow", "11-maxflowapps", "12-nphard",
            ]
        ],
        "file_prefix": "algorithms_erickson",
    },
    {
        "category": "Theory of Computation",
        "title": "Models of Computation",
        "authors": "Jeff Erickson",
        "license": "CC BY 4.0",
        "kind": "pdf",
        "pages": [
            {"name": n, "url": f"https://jeffe.cs.illinois.edu/teaching/algorithms/models/{n}.pdf"}
            for n in [
                "01-strings", "02-regular", "03-automata", "04-nfa", "05-context-free",
                "06-turing-machines", "07-undecidable", "08-universal", "09-nondeterminism",
            ]
        ],
        "file_prefix": "toc_erickson",
    },
    {
        "category": "Machine Learning",
        "title": "A Course in Machine Learning",
        "authors": "Hal Daumé III",
        "license": "Creative Commons",
        "kind": "pdf",
        "pages": [
            {"name": f"ch{n:02d}", "url": f"http://ciml.info/dl/v0_99/ciml-v0_99-ch{n:02d}.pdf"}
            for n in range(0, 20)
        ],
        "file_prefix": "ml_a_course_in_machine_learning",
    },
    {
        "category": "Operating Systems",
        "title": "Think OS: A Brief Introduction to Operating Systems",
        "authors": "Allen B. Downey",
        "license": "CC BY-NC-SA 4.0",
        "kind": "pdf",
        "pages": [{"name": "think_os", "url": "https://greenteapress.com/thinkos/thinkos.pdf"}],
        "file_prefix": "os_think_os",
    },
    {
        "category": "Computer Networking",
        "title": "An Introduction to Computer Networks",
        "authors": "Peter L. Dordal",
        "license": "CC BY-NC-ND 3.0",
        "kind": "pdf",
        "pages": [
            {"name": "intro_to_computer_networks", "url": "https://intronetworks.cs.luc.edu/current2/ComputerNetworks.pdf"}
        ],
        "file_prefix": "net_intro_to_computer_networks",
    },
    {
        "category": "Compilers",
        "title": "Basics of Compiler Design",
        "authors": "Torben Æ. Mogensen",
        "license": "CC BY-NC-ND 3.0",
        "kind": "pdf",
        "pages": [
            {"name": "basics_of_compiler_design", "url": "https://hjemmesider.diku.dk/~torbenm/Basics/basics_lulu2.pdf"}
        ],
        "file_prefix": "compilers_basics_of_compiler_design",
    },
    {
        "category": "Discrete Mathematics",
        "title": "Discrete Mathematics: An Open Introduction (3rd ed)",
        "authors": "Oscar Levin",
        "license": "CC BY-SA 4.0",
        "kind": "pdf",
        "pages": [
            {"name": "open_intro", "url": "https://discrete.openmathbooks.org/pdfs/dmoi3-tablet.pdf"}
        ],
        "file_prefix": "discrete_math_open_intro",
    },
]


def slugify(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


def write_md(filename: str, text: str) -> str:
    target = NOTES_DIR / f"{filename}.md"
    target.write_text(text, encoding="utf-8")
    return target.name


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def pdf_to_md(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts = [(i, page.get_text("text")) for i, page in enumerate(doc, start=1)]
    doc.close()
    cleaned = clean_page_texts(page_texts)
    return "\n\n".join(f"[[PAGE {i}]]\n\n{text}" for i, text in cleaned)


def handle_pdf(source: dict, manifest: list[dict]) -> None:
    pdf_dir = NOTES_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    def fetch(page: dict) -> tuple[str, str, str]:
        pdf = http_get(page["url"])
        md = pdf_to_md(pdf)
        md = f"# {page['name']}\n\nSource: {page['url']}\n\n" + md
        prefix = f"{source['file_prefix']}_{slugify(page['name'])}"
        pdf_filename = f"{prefix}.pdf"
        (pdf_dir / pdf_filename).write_bytes(pdf)
        return prefix, md, pdf_filename

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, page): page for page in source["pages"]}
        for future in as_completed(futures):
            page = futures[future]
            try:
                prefix, md, pdf_filename = future.result()
            except Exception as e:
                print(f"  ! failed {page['url']}: {e}")
                continue
            filename = write_md(prefix, md)
            manifest.append(
                {
                    "id": prefix,
                    "title": f"{source['title']} - {page['name']}",
                    "authors": [source["authors"]],
                    "category": source["category"],
                    "license": source.get("license", ""),
                    "local_path": f"{NOTES_DIR.relative_to(PROJECT_ROOT)}/{filename}",
                    "pdf_path": f"{NOTES_DIR.relative_to(PROJECT_ROOT)}/pdfs/{pdf_filename}",
                    "url": page["url"],
                }
            )
            print(f"  + {filename} (pdf: {pdf_filename})")


def main() -> None:
    if NOTES_DIR.exists():
        shutil.rmtree(NOTES_DIR)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for source in SOURCES:
        print(f"\n--- {source['category']}: {source['title']} ---")
        try:
            if source["kind"] == "pdf":
                handle_pdf(source, manifest)
            else:
                print(f"  ! unknown source kind: {source['kind']}")
        except Exception as e:
            print(f"  ! source failed: {e}")
            continue

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(f"\nFetched {len(manifest)} note files into {NOTES_DIR}")


if __name__ == "__main__":
    main()