"""One-shot production fix script — removes after successful run."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent.parent  # code_review_agent/

# ── Fix 1: Dockerfile ───────────────────────────────────────────────────────
dockerfile = ROOT / "probe" / "server" / "Dockerfile"
lines = dockerfile.read_text(encoding="utf-8").splitlines(keepends=True)

# Strip Meta copyright header (lines before "# Multi-stage build")
start = next(
    i for i, l in enumerate(lines) if l.strip().startswith("# Multi-stage build")
)
content = "".join(lines[start:])

# Fix stale CMD module path
content = content.replace(
    "uvicorn server.app:app",
    "uvicorn probe.server.app:app",
)

dockerfile.write_text(content, encoding="utf-8")
print("[OK] Dockerfile — copyright header removed, CMD path updated to probe.server.app:app")

# ── Fix 2: pyproject.toml ───────────────────────────────────────────────────
pyproject = ROOT / "probe" / "pyproject.toml"
text = pyproject.read_bytes()

# Repair garbled UTF-8 em-dash (â€" = U+2014 encoded as latin-1 over UTF-8)
text = text.replace(
    "PRobe \xe2\x80\x94 Pull Request".encode("utf-8"),
    "PRobe \u2014 Pull Request".encode("utf-8"),
)
content = text.decode("utf-8")

# Remove duplicate [dependency-groups] block (keep [project.optional-dependencies])
if "[dependency-groups]" in content:
    idx = content.rfind("[dependency-groups]")
    content = content[:idx].rstrip() + "\n"
    print("[OK] pyproject.toml — duplicate [dependency-groups] block removed")

pyproject.write_text(content, encoding="utf-8")
print("[OK] pyproject.toml — description encoding fixed")

print("\nAll fixes applied.")
