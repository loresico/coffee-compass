# Contributing Guide

Thank you for considering contributing to this project! This guide will help you understand our development process and standards.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style](#code-style)

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intentions

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/python-uv-template.git
cd python-uv-template

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/python-uv-template.git
```

### 2. Create a Branch

```bash
# Always create a new branch for your changes
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 3. Make Your Changes

Follow the project structure and coding standards (see below).

## 📝 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type

Must be one of:

- **feat**: New feature for the user
- **fix**: Bug fix for the user
- **docs**: Documentation only changes
- **style**: Changes that don't affect code meaning (formatting, whitespace)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to build process or auxiliary tools
- **ci**: Changes to CI configuration files and scripts
- **revert**: Reverts a previous commit

### Scope (Optional)

The scope should specify the place of the commit change:

- `setup` - Changes to setup-portable.sh
- `package` - Changes to package-portable.sh
- `deps` - Dependency updates
- `docs` - Documentation
- `config` - Configuration files
- `ci` - CI/CD related

### Subject

- Use imperative, present tense: "add" not "added" nor "adds"
- Don't capitalize the first letter
- No period (.) at the end
- Maximum 50 characters

### Body (Optional)

- Use imperative, present tense
- Include motivation for the change
- Contrast with previous behavior

### Footer (Optional)

- Reference issues: `Closes #123`, `Fixes #456`
- Note breaking changes: `BREAKING CHANGE: description`

### Examples

#### Simple Feature
```
feat(setup): add Python 3.14 support
```

#### Bug Fix with Details
```
fix(package): correct tar.gz extraction path

The extraction was failing on macOS because of incorrect
strip-components value. Changed from 2 to 1.

Fixes #42
```

#### Documentation Update
```
docs(readme): update installation instructions

Added clarification about Python version compatibility
and release date matching.
```

#### Breaking Change
```
feat(setup): change default Python to 3.13

BREAKING CHANGE: Default Python version changed from 3.12 to 3.13.
Users need to run `./setup-portable.sh --force-clean` to upgrade.

Closes #89
```

#### Dependency Update
```
chore(deps): upgrade uv to 0.5.0

- Updated uv from 0.4.x to 0.5.0
- Includes performance improvements
- No breaking changes
```

#### Refactoring
```
refactor(setup): improve error handling

Extracted error handling into separate functions
for better readability and reusability.
```

## 🔄 Pull Request Process

### 1. Sync with Upstream

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Run Tests and Checks

```bash
# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Test the setup script
./setup-portable.sh --force-clean
```

### 3. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 4. Create Pull Request

Go to GitHub and create a Pull Request with:

**Title:** Follow commit message convention
```
feat(setup): add Python 3.14 support
```

**Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
Describe how you tested your changes

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] I have updated the documentation accordingly
- [ ] My commits follow the conventional commits specification

## Related Issues
Closes #(issue number)
```

### 5. Code Review

- Be open to feedback
- Respond to comments promptly
- Make requested changes in new commits
- Don't force-push after review has started

## 🛠️ Development Setup

### Prerequisites

- Python 3.13+
- UV package manager
- Git

### Setup Development Environment

```bash
# Run setup
./setup-portable.sh

# Activate virtual environment
source .venv/bin/activate

# Install dev dependencies
uv pip install -e ".[dev]"
```

### Project Structure

```
.
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_main.py
├── setup-portable.sh       # Setup script
├── package-portable.sh     # Packaging script
├── verify-python-version.sh # Verification script
├── pyproject.toml          # Project configuration
└── README.md              # Documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest -v

# Run and show print statements
pytest -s
```

### Writing Tests

- Use `pytest` framework
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive names
- Test edge cases
- Add docstrings

Example:
```python
def test_greet_with_custom_name():
    """Test greet function with a custom name."""
    result = greet("Alice")
    assert result == "Hello, Alice!"
```

## 🎨 Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for formatting (88 char line length)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate
- Write docstrings for functions and classes

```python
def greet(name: str) -> str:
    """
    Return a greeting message.
    
    Args:
        name: The name to greet
        
    Returns:
        A greeting string
    """
    return f"Hello, {name}!"
```

### Bash Scripts

- Use `shellcheck` for linting
- Use `set -e` for error handling
- Add comments for complex logic
- Use meaningful variable names
- Quote variables: `"$var"` not `$var`

```bash
#!/usr/bin/env bash
set -e

# Configuration
PYTHON_VERSION="3.13.9"

echo "Setting up Python ${PYTHON_VERSION}"
```

### Documentation

- Use Markdown for documentation
- Include code examples
- Keep it concise and clear
- Update README when adding features
- Add comments for complex code

## 📦 Release Process

(For Maintainers)

### Version Bump

```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit with conventional commit
git commit -m "chore(release): bump version to 1.0.0"
```

### Tagging

```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## 🐛 Reporting Bugs

### Before Reporting

- Check existing issues
- Try latest version
- Verify it's reproducible

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run '...'
2. See error

**Expected behavior**
What you expected to happen

**Environment**
- OS: [e.g., macOS 13.0, Ubuntu 22.04]
- Python version: [e.g., 3.13.9]
- Script output: [paste here]

**Additional context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem

**Describe the solution you'd like**
A clear description of what you want to happen

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other context or screenshots
```

## 📞 Questions?

- Open a [Discussion](https://github.com/username/repo/discussions)
- Check the [README](README.md)
- Review [existing issues](https://github.com/username/repo/issues)

## 🙏 Thank You!

Your contributions make this project better for everyone!

---

**Happy Contributing! 🎉**