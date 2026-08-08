"""Fetch a curated set of open undergraduate computer science notes into ./csc_notes.

Each source is normalized to a Markdown file under csc_notes/ and registered in
csc_notes/library_metadata.json so it can be indexed by scripts/index_library.py.

Sources:
  - Crafting Interpreters            (Compilers, Markdown from GitHub)
  - CNP3 2nd ed.                     (Computer Networking, reStructuredText from GitHub)
  - Computer Science from the Bottom Up (OS/Computer Architecture, HTML)
  - Open Data Structures             (Data Structures, PDF)
  - Algorithms (Jeff Erickson)       (Algorithms, PDF)
  - Models of Computation (Erickson) (Theory of Computation, PDF)
  - Discrete Mathematics (Levin)     (Discrete Math, HTML)
  - SQL for Web Nerds                (Databases, HTML)
  - Software Engineering at Google   (Software Engineering, HTML)
  - A Course in Machine Learning     (Intro ML, PDF)
"""

import io
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import html2text
from docutils.core import publish_parts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = PROJECT_ROOT / "csc_notes"
MANIFEST_PATH = NOTES_DIR / "library_metadata.json"
STAGING = PROJECT_ROOT / ".fetch_notes_tmp"

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

SKIP_FILES = {"contents.md", "index.md", "acknowledgements.md", "backmatter.md", "dedication.md", "welcome.md"}

SOURCES = [
    {
        "category": "Compilers",
        "title": "Crafting Interpreters",
        "authors": "Robert Nystrom",
        "license": "CC BY-NC-SA 4.0",
        "kind": "github_md",
        "repo": "munificent/craftinginterpreters",
        "repo_paths": ["book"],
        "url_base": "https://github.com/munificent/craftinginterpreters/blob/master/book",
        "file_prefix": "compilers_crafting_interpreters",
        "skip": SKIP_FILES,
    },
    {
        "category": "Computer Networking",
        "title": "Computer Networking: Principles, Protocols and Practice (2nd ed)",
        "authors": "Olivier Bonaventure et al.",
        "license": "CC BY 4.0",
        "kind": "github_rst",
        "repo": "obonaventure/cnp3",
        "repo_paths": ["book-2nd/principles", "book-2nd/protocols"],
        "url_base": "https://github.com/obonaventure/cnp3/blob/master/book-2nd",
        "file_prefix": "networking_cnp3",
    },
    {
        "category": "Operating Systems",
        "title": "Computer Science from the Bottom Up",
        "authors": "Ian Wienand",
        "license": "CC BY-SA 3.0",
        "kind": "html",
        "url_base": "https://www.bottomupcs.com/",
        "pages": [
            "pf01", "ch01", "ch01s02", "ch01s03", "ch02", "ch02s02",
            "ch03", "ch03s02", "ch03s03", "ch03s04",
            "ch04", "ch04s02", "ch04s03", "ch04s04",
            "ch05", "ch05s02", "ch05s03", "ch05s04", "ch05s05", "ch05s06", "ch05s07", "ch05s08",
            "ch06", "ch06s02", "ch06s03", "ch06s04", "ch06s05", "ch06s06", "ch06s07", "ch06s08", "ch06s09", "ch06s10",
            "ch07", "ch07s02", "ch07s03", "ch07s04", "ch07s05", "ch07s06",
            "ch08", "ch08s02", "ch08s03", "ch08s04", "ch08s05", "ch08s06", "ch08s07",
            "ch09", "ch09s02", "ch09s03", "ch09s04", "ch09s05",
        ],
        "file_prefix": "os_bottomupcs",
    },
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
        "category": "Discrete Mathematics",
        "title": "Discrete Mathematics: An Open Introduction (3rd ed)",
        "authors": "Oscar Levin",
        "license": "CC BY-SA 4.0",
        "kind": "html",
        "url_base": "https://discrete.openmathbooks.org/dmoi3/",
        "pages": ["frontmatter", "preface", "ch_intro", "ch_logic", "ch_counting", "ch_sequences", "ch_graphtheory"],
        "file_prefix": "discrete_math_levin",
    },
    {
        "category": "Databases",
        "title": "SQL for Web Nerds",
        "authors": "Philip Greenspun",
        "license": "CC BY-SA 4.0",
        "kind": "html",
        "url_base": "https://philip.greenspun.com/sql/",
        "pages": [
            "preface", "introduction", "data-modeling", "queries", "complex-queries",
            "transactions", "triggers", "views", "style", "procedural", "trees", "dates",
            "limits", "tuning", "data-warehousing", "foreign-and-legacy-data", "afterword",
            "your-own-rdbms", "getting-real-work-done",
        ],
        "file_prefix": "db_sql_for_web_nerds",
    },
    {
        "category": "Software Engineering",
        "title": "Software Engineering at Google",
        "authors": "Titus Winters, Tom Manshreck, Hyrum Wright",
        "license": "CC BY-NC-ND 4.0",
        "kind": "html",
        "url_base": "https://abseil.io/resources/swe-book/html/",
        "pages": [f"ch{i:02d}" for i in range(1, 27)],
        "file_prefix": "swe_at_google",
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
]


def slugify(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


def write_md(filename: str, text: str) -> None:
    target = NOTES_DIR / f"{filename}.md"
    target.write_text(text, encoding="utf-8")
    return target.name


def http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def html_to_md(html: str) -> str:
    return html2text.html2text(html)


def rst_to_md(rst: str) -> str:
    html = publish_parts(
        rst,
        writer_name="html",
        settings_overrides={"report_level": 5, "halt_level": 5, "warning_stream": io.StringIO()},
    )["body"]
    return html_to_md(html)


def pdf_to_md(pdf_bytes: bytes, source_url: str) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"[[PAGE {i}]]\n\n{text}")
    doc.close()
    return "\n\n".join(pages)


def clone_sparse(repo: str, paths: list[str]) -> Path:
    STAGING.mkdir(exist_ok=True)
    target = STAGING / repo.replace("/", "_")
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", f"https://github.com/{repo}.git", str(target)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", *paths],
        cwd=target,
        check=True,
        capture_output=True,
    )
    return target


def handle_github(source: dict, manifest: list[dict]) -> None:
    paths = source["repo_paths"]
    target = clone_sparse(source["repo"], paths)
    ext = "*.md" if source["kind"] == "github_md" else "*.rst"
    skip = source.get("skip", set())
    used_names: set[str] = set()

    for root in [target / p for p in paths]:
        files = sorted(root.glob(ext))
        if source["kind"] == "github_rst":
            files = [p for p in files if p.parent.name in ("principles", "protocols")]

        for path in files:
            if path.name in skip:
                continue
            rel = path.relative_to(target)
            prefix = f"{source['file_prefix']}_{slugify(rel.stem)}"
            if prefix in used_names:
                prefix = f"{prefix}_{len(used_names)}"
            used_names.add(prefix)
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if source["kind"] == "github_rst":
                try:
                    raw = rst_to_md(raw)
                except Exception:
                    raw = re.sub(r"^=+\s*$|^-+\s*$", "", raw, flags=re.MULTILINE)
            filename = write_md(prefix, raw)
            manifest.append(
                {
                    "id": prefix,
                    "title": f"{source['title']} - {rel.stem}",
                    "authors": [source["authors"]],
                    "category": source["category"],
                    "license": source.get("license", ""),
                    "local_path": f"{NOTES_DIR.relative_to(PROJECT_ROOT)}/{filename}",
                    "url": f"{source['url_base']}/{rel}",
                }
            )
            print(f"  + {filename}")


def handle_html(source: dict, manifest: list[dict]) -> None:
    def fetch(page: str) -> tuple[str, str]:
        url = f"{source['url_base']}{page}.html"
        html = http_get(url).decode("utf-8", errors="ignore")
        md = html_to_md(html)
        md = f"# {page}\n\nSource: {url}\n\n" + md
        prefix = f"{source['file_prefix']}_{slugify(page)}"
        return prefix, md

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, page): page for page in source["pages"]}
        for future in as_completed(futures):
            page = futures[future]
            try:
                prefix, md = future.result()
            except Exception as e:
                print(f"  ! failed {page}: {e}")
                continue
            filename = write_md(prefix, md)
            manifest.append(
                {
                    "id": prefix,
                    "title": f"{source['title']} - {page}",
                    "authors": [source["authors"]],
                    "category": source["category"],
                    "license": source.get("license", ""),
                    "local_path": f"{NOTES_DIR.relative_to(PROJECT_ROOT)}/{filename}",
                    "url": f"{source['url_base']}{page}.html",
                }
            )
            print(f"  + {filename}")


def handle_pdf(source: dict, manifest: list[dict]) -> None:
    pdf_dir = NOTES_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    def fetch(page: dict) -> tuple[str, str, str]:
        pdf = http_get(page["url"])
        md = pdf_to_md(pdf, page["url"])
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
            if source["kind"] in ("github_md", "github_rst"):
                handle_github(source, manifest)
            elif source["kind"] == "html":
                handle_html(source, manifest)
            elif source["kind"] == "pdf":
                handle_pdf(source, manifest)
        except Exception as e:
            print(f"  ! source failed: {e}")
            continue

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    if STAGING.exists():
        shutil.rmtree(STAGING)

    print(f"\nFetched {len(manifest)} note files into {NOTES_DIR}")


if __name__ == "__main__":
    main()
