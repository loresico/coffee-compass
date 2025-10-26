# Python UV Template (Always Portable) 🚀

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-enabled-blue)](https://github.com/astral-sh/uv)
[![Portable](https://img.shields.io/badge/portable-100%25-green)](https://github.com/indygreg/python-build-standalone)

A professional Python project template that **always uses portable Python** - never relies on system installations.

## ✨ Philosophy

- 🎯 **Always Portable** - Consistent Python environment everywhere
- 📦 **Self-Contained** - No dependency on system Python
- 🔄 **Reproducible** - Same exact Python version every time
- 🚀 **Distributable** - Package includes everything needed

## 🚀 Quick Start

```bash
# 1. Clone or use this template
git clone https://github.com/yourusername/your-project
cd your-project

# 2. Run setup (downloads Python 3.13.9 first time, ~2 minutes)
./setup.sh

# 3. Activate and run
source .venv/bin/activate
python src/main.py
```

That's it! No system Python needed.

## 📁 Project Structure (After Setup)

```
your-project/
├── .github/
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline
├── .python/                  # Portable Python 3.13.9 (~80MB, gitignored)
├── .venv/                    # Virtual environment (gitignored)
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── .gitignore
├── CONTRIBUTING.md           # Contribution guidelines
├── pyproject.toml            # Project configuration
├── README.md
├── QUICK_REFERENCE.md
├── setup.sh         # Setup script
├── uv.lock
└── verify-python-version.sh  # Version checker
```

## 🎯 Common Commands

```bash
# Setup (first time downloads Python, subsequent runs reuse it)
./setup.sh

# If something breaks, clean rebuild
./setup.sh --force-clean

# Add a dependency
# 1. Edit pyproject.toml
# 2. Then:
uv lock
uv sync

```

## 🔧 How It Works

### First Time

1. Downloads pre-built Python 3.13.9 from [python-build-standalone](https://github.com/indygreg/python-build-standalone)
2. Installs to `.python/` directory
3. Creates virtual environment in `.venv/`
4. Installs dependencies with UV

### Subsequent Runs

1. Finds existing `.python/` installation
2. Reuses it (no download needed!)
3. Creates fresh `.venv/`
4. Installs dependencies

### With `--force-clean`

1. Deletes `.python/`, `.venv/`, `uv.lock`
2. Downloads Python 3.13.9 again
3. Fresh installation

## 💡 Why Portable Python?

| System Python | Portable Python |
|---------------|-----------------|
| ❌ Different versions on different machines | ✅ Exact same version everywhere |
| ❌ Might not be installed | ✅ Always available |
| ❌ User might update it | ✅ Controlled version |
| ❌ Dependency conflicts | ✅ Self-contained |
| ❌ "Works on my machine" | ✅ Works everywhere |

## 📊 Disk Space

- `.python/` : ~80 MB (one-time)
- `.venv/` : ~50 MB (varies by dependencies)
- **Total**: ~150 MB uncompressed
- **Package**: ~50 MB compressed

Small price for complete portability!

## 🛠️ Development Workflow

```bash
# Day 1: Setup
./setup.sh
source .venv/bin/activate

# Daily development
python src/main.py
pytest

# Add dependencies
# Edit pyproject.toml, then:
uv lock && uv sync

# If weird issues
./setup.sh --force-clean
```

## 📮 Distribution Workflow

```bash
# 1. Ensure clean build
./setup.sh --force-clean

# 2. Test your app
source .venv/bin/activate
python src/main.py
```

## 🎨 Customization

### Change Python Version

Edit `setup.sh`:
```bash
PYTHON_VERSION="3.14.*"  # Or any version
```

Available versions: https://github.com/indygreg/python-build-standalone/releases

## 🐛 Troubleshooting

```bash
# Virtual environment issues
rm -rf .venv/ && ./setup.sh

# Complete fresh start
./setup.sh --force-clean

# Check what you have
.python/bin/python3 --version
source .venv/bin/activate && python --version
```

## 📚 Documentation

- [Setup Guide](README.md) - Detailed guide
- [Quick Reference](QUICK_REFERENCE.md) - Command cheat sheet
- [UV Documentation](https://github.com/astral-sh/uv)

## ⚠️ Platform Compatibility

Portable Python is **OS and architecture specific**:

- ✅ macOS x86_64 → macOS x86_64
- ✅ macOS arm64 (M1/M2/M3) → macOS arm64
- ✅ Linux x86_64 → Linux x86_64
- ✅ Linux aarch64 → Linux aarch64
- ❌ macOS → Linux (use Docker)
- ❌ x86_64 → arm64 (use Docker)
- ❌ Windows (use WSL or Docker)

## 🤝 Contributing

### Commit Message Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for clear and structured commit history.

**Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature change or bug fix)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, config)
- `ci`: CI/CD changes

**Examples:**
```bash
feat(setup): add Python 3.14 support
fix(package): correct tar.gz extraction path
docs(readme): update installation instructions
chore(deps): upgrade uv to latest version
refactor(setup): improve error handling
```

**Scope (optional):** Component affected (setup, package, docs, etc.)

### Pull Request Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with conventional commits
4. Run tests and linting (`pytest && black --check .`)
5. Update documentation if needed
6. Commit your changes (`git commit -m 'feat: add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request with:
   - Clear description of changes
   - Link to related issues
   - Screenshots/examples if applicable

### Code Standards

- **Python Code:** Follow PEP 8, use Black formatter
- **Bash Scripts:** Use ShellCheck for validation
- **Documentation:** Clear, concise, with examples
- **Tests:** Add tests for new features

### Before Submitting

```bash
# Format code
black src/ tests/

# Run linting
flake8 src/ tests/

# Run tests
pytest tests/

# Verify script works
./setup.sh --force-clean
```

### Set Up Commit Message Template (Optional)

```bash
# Use the included commit message template
git config commit.template .gitmessage

# Now when you commit, you'll see helpful hints
git commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [python-build-standalone](https://github.com/indygreg/python-build-standalone) - Pre-built Python distributions
- [UV](https://github.com/astral-sh/uv) - Fast Python package installer
- Built with modern Python best practices

## 📧 Support

- 📖 [Documentation](SETUP_GUIDE.md)
- 🐛 [Issues](https://github.com/yourusername/python-uv-template/issues)
- 💬 [Discussions](https://github.com/yourusername/python-uv-template/discussions)

---

⭐ If you find this template helpful, please star it!

**Ready to use?** Click "Use this template" above or clone and start building!
