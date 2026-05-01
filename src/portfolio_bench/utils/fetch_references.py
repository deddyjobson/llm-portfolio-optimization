"""Download reference papers and repositories."""

import subprocess
from pathlib import Path

import requests

# Reference paper URLs
PAPERS = {
    "olps_toolbox": "https://jmlr.csail.mit.edu/papers/volume17/15-317/15-317.pdf",  # OLPS toolbox JMLR
    "cvar_optimization": "https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf",  # Rockafellar & Uryasev
    "chain_of_thought": "https://arxiv.org/pdf/2201.11903.pdf",
    "few_shot_learners": "https://arxiv.org/pdf/2005.14165.pdf",
}

# Reference repositories
REPOS = {
    "olps": "https://github.com/OLPS/OLPS.git",
    "universal_portfolios": "https://github.com/Marigold/universal-portfolios.git",
}


def download_paper(url: str, output_path: Path) -> bool:
    """Download a PDF paper if it doesn't already exist.

    Args:
        url: URL of the PDF.
        output_path: Path to save the PDF.

    Returns:
        True if successful (or already exists), False otherwise.
    """
    # Idempotent: skip if file already exists
    if output_path.exists():
        print(f"  {output_path.name} already exists, skipping")
        return True

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def clone_repo(url: str, output_dir: Path) -> bool:
    """Shallow clone a git repository if it doesn't already exist.

    Args:
        url: Repository URL.
        output_dir: Directory to clone into.

    Returns:
        True if successful (or already exists), False otherwise.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_name = url.split("/")[-1].replace(".git", "")
        dest = output_dir / repo_name

        # Idempotent: skip if repo already exists
        if dest.exists():
            print(f"  Repository {repo_name} already exists, skipping")
            return True

        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
        )
        return True
    except Exception as e:
        print(f"Failed to clone {url}: {e}")
        return False


def fetch_all_references(output_dir: str = "references") -> None:
    """Download all reference papers and clone repositories.

    Args:
        output_dir: Base output directory.
    """
    output_path = Path(output_dir)
    papers_dir = output_path / "papers"
    repos_dir = output_path / "repos"

    # Download papers
    print("Downloading reference papers...")
    for name, url in PAPERS.items():
        pdf_path = papers_dir / f"{name}.pdf"
        print(f"  Downloading {name}...")
        download_paper(url, pdf_path)

    # Clone repositories
    print("\nCloning reference repositories...")
    for name, url in REPOS.items():
        print(f"  Cloning {name}...")
        clone_repo(url, repos_dir)

    print(f"\nReferences saved to: {output_path}")
