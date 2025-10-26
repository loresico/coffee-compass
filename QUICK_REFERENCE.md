# Quick Reference Card

## 🚀 Common Commands

```bash
# Setup (uses existing or downloads portable Python)
./setup.sh

# Clean rebuild (removes everything, starts fresh)
./setupsh --force-clean

# Show help
./setup.sh --help
```

## 📊 Quick Decision Guide

```
First time?
└─ ./setup-portable.sh

Something broken?
└─ ./setup-portable.sh --force-clean

Changed dependencies?
├─ Edit pyproject.toml
└─ uv lock && uv sync
```

## 🎯 Common Workflows

| Task | Commands |
|------|----------|
| **Initial setup** | `./setup-portable.sh` |
| **Daily development** | `source .venv/bin/activate` → `python src/main.py` |
| **Add dependency** | Edit `pyproject.toml` → `uv lock` → `uv sync` |
| **Fix issues** | `./setup-portable.sh --force-clean` |
| **Complete reset** | `./setup-portable.sh --force-clean` |

## 📁 What Gets Created

```
.python/          # Portable Python (~80MB, gitignored)
.venv/            # Virtual environment (~50MB, gitignored)
uv.lock           # Locked dependencies (committed)
dist/             # Packages (~50MB compressed, gitignored)
```

## 🎨 Color Guide

- ✅ Green = Success, found existing
- 📥 Blue = Downloading/installing
- 🧹 Yellow = Cleaning
- ❌ Red = Error
- 💡 Light blue = Info/tips

## ⚡ Speed Guide

| Operation | Time | Network |
|-----------|------|---------|
| First setup (download) | 1-2 min | Required |
| Reuse existing Python | 10-30 sec | Optional |
| Clean rebuild | 1-2 min | Required |

## 💾 Disk Space

| Item | Size |
|------|------|
| Portable Python | ~80 MB |
| Virtual environment | ~50 MB |
| Total unpackaged | ~150 MB |
| Packaged (tar.gz) | ~50 MB |

## 🚫 .gitignore

Always ignore these:
```gitignore
.python/
.venv/
dist/
uv.lock          # Optional: commit for reproducibility
```

## 🔧 Quick Fixes

```bash
# Virtual environment broken
rm -rf .venv/ && ./setup-portable.sh

# Everything broken
./setup-portable.sh --force-clean

# UV not working
source .venv/bin/activate && pip install --upgrade uv

# Dependencies out of sync
uv lock && uv sync
```

## 🎯 Key Principles

1. **Always portable** - Never uses system Python
2. **Reuse when possible** - Keeps existing `.python/` if found
3. **Fresh venv** - Always recreates `.venv/`
4. **Clean on demand** - `--force-clean` for fresh start

## 📞 Need Help?

```bash
# Show help
./setup-portable.sh --help

# Check versions
.python/bin/python3 --version
source .venv/bin/activate && python --version

# Debug
ls -la .python/bin/
ls -la .venv/bin/
```
