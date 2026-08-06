from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "app.py",
    "enterprise_analytics.py",
    "demand_business_analytics_fixed.py",
    "zero_shot_demand_mvp_core_generic_v2.py",
    "ornek_gecmis_satis_v4.csv",
    "ornek_gelecek_stok_plani_v4.csv",
]

missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
if missing:
    raise SystemExit(f"Eksik dosyalar: {missing}")

history = pd.read_csv(ROOT / "ornek_gecmis_satis_v4.csv")
plan = pd.read_csv(ROOT / "ornek_gelecek_stok_plani_v4.csv")

if history.empty or plan.empty:
    raise SystemExit("Örnek veri dosyalarından biri boş.")

required_history = {
    "Tarih", "Magaza_ID", "Urun_ID",
    "Satis_Adedi", "Stok_Miktari",
}
required_plan = {
    "Tarih", "Magaza_ID", "Urun_ID",
    "Baslangic_Stoku", "Planlanan_Sevkiyat",
}
if not required_history.issubset(history.columns):
    raise SystemExit(
        f"Geçmiş örnek veride eksik sütunlar: "
        f"{sorted(required_history - set(history.columns))}"
    )
if not required_plan.issubset(plan.columns):
    raise SystemExit(
        f"Plan örnek veride eksik sütunlar: "
        f"{sorted(required_plan - set(plan.columns))}"
    )

secret_patterns = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
]

scan_extensions = {".py", ".md", ".txt", ".toml", ".yml", ".yaml"}
violations = []
for path in ROOT.rglob("*"):
    if (
        not path.is_file()
        or path.suffix.lower() not in scan_extensions
        or ".git" in path.parts
    ):
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pattern in secret_patterns:
        if pattern.search(text):
            violations.append(str(path.relative_to(ROOT)))

if violations:
    raise SystemExit(
        "Olası secret/token bulunan dosyalar: "
        + ", ".join(sorted(set(violations)))
    )

print(
    "Repo doğrulaması başarılı:",
    f"{len(history):,} geçmiş satır,",
    f"{len(plan):,} plan satırı.",
)
