#!/usr/bin/env python3
"""Fetch legal free theory PDFs + extract text for offline reading."""

from __future__ import annotations

import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "pdfs"
EXT = ROOT / "extracted"

BOOKS = {
    "RLbook2018.pdf": "http://incompleteideas.net/book/RLbook2020.pdf",
    "Bertsekas-RLCOURSECOMPLETE_2ndED.pdf": (
        "https://web.mit.edu/dimitrib/www/RLCOURSECOMPLETE%202ndEDITION.pdf"
    ),
    "Bertsekas-LessonsfromAlphazero.pdf": (
        "https://web.mit.edu/dimitrib/www/LessonsfromAlphazero.pdf"
    ),
    "Bertsekas-Rollout_Complete_Book.pdf": (
        "https://web.mit.edu/dimitrib/www/Rollout_Complete%20Book.pdf"
    ),
    "Bertsekas-Abstract_DP_RL_Lecture.pdf": (
        "http://web.mit.edu/dimitrib/www/Abstract_DP_RL_Lecture.pdf"
    ),
    "Bertsekas-DP_Slides_2015.pdf": (
        "https://web.mit.edu/dimitrib/www/dpchapter.pdf"
    ),
    "NgHaradaRussell-shaping-ICML1999.pdf": (
        "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/"
        "NgHaradaRussell-shaping-ICML1999.pdf"
    ),
    "SuttonPrecupSingh-options-AIJ1999.pdf": (
        "https://people.cs.umass.edu/~barto/courses/cs687/"
        "Sutton-Precup-Singh-AIJ99.pdf"
    ),
    "RossGordonBagnell-DAgger.pdf": "https://arxiv.org/pdf/1011.0686.pdf",
    "AnthonyTianBarber-ExIt.pdf": "https://arxiv.org/pdf/1705.08439.pdf",
    "Silver-AlphaZero.pdf": "https://arxiv.org/pdf/1712.01815.pdf",
    "Ecoffet-GoExplore.pdf": "https://arxiv.org/pdf/2004.12919.pdf",
}

# Sutton chapter PDF-page ranges (approx; 2nd ed. incompleteideas build)
SUTTON_RANGES = {
    "sutton_ch1_2.txt": (20, 50),
    "sutton_ch3_4.txt": (45, 100),
    "sutton_ch8.txt": (155, 200),
    "sutton_ch16_17.txt": (450, 520),
}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"skip {dest.name}")
        return
    print(f"get  {dest.name}")
    urllib.request.urlretrieve(url, dest)


def pdftotext(src: Path, dest: Path, fr: int | None = None, to: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["pdftotext", "-layout"]
    if fr is not None:
        cmd += ["-f", str(fr), "-l", str(to)]
    cmd += [str(src), str(dest)]
    subprocess.check_call(cmd)


def main() -> None:
    PDF.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    for name, url in BOOKS.items():
        try:
            fetch(url, PDF / name)
        except Exception as e:  # noqa: BLE001 — keep going on one bad URL
            print(f"FAIL {name}: {e}")

    sutton = PDF / "RLbook2018.pdf"
    if sutton.exists():
        for out, (a, b) in SUTTON_RANGES.items():
            pdftotext(sutton, EXT / out, a, b)

    for name in BOOKS:
        src = PDF / name
        if not src.exists() or name == "RLbook2018.pdf":
            continue
        if name.startswith("Bertsekas-") and "Lessons" not in name and "Abstract" not in name:
            # large books: front + early body only by default
            continue
        pdftotext(src, EXT / f"{Path(name).stem}.txt")

    # Bertsekas Lessons + RL course key chunks
    az = PDF / "Bertsekas-LessonsfromAlphazero.pdf"
    if az.exists():
        pdftotext(az, EXT / "bertsekas_az_front.txt", 1, 30)
        pdftotext(az, EXT / "bertsekas_az_ch1_2.txt", 20, 80)
        pdftotext(az, EXT / "bertsekas_az_mid.txt", 80, 160)
        pdftotext(az, EXT / "bertsekas_az_end.txt", 160, 242)
    course = PDF / "Bertsekas-RLCOURSECOMPLETE_2ndED.pdf"
    if course.exists():
        pdftotext(course, EXT / "bertsekas_rlcourse_front.txt", 1, 40)
        pdftotext(course, EXT / "bertsekas_rlcourse_early.txt", 40, 120)
        pdftotext(course, EXT / "bertsekas_rlcourse_approx.txt", 120, 220)
    rollout = PDF / "Bertsekas-Rollout_Complete_Book.pdf"
    if rollout.exists():
        pdftotext(rollout, EXT / "bertsekas_rollout_front.txt", 1, 30)
        pdftotext(rollout, EXT / "bertsekas_rollout_early.txt", 40, 120)

    print("done. Puterman draft: git clone https://github.com/martyput/MDP_book")
    print("  into notes/theory/pdfs/puterman_mdp_draft/ if needed.")


if __name__ == "__main__":
    main()
