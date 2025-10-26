#!/usr/bin/env bash
set -e

# Configuration
# IMPORTANT: Python version and release date MUST match!
# Check available combinations: https://github.com/astral-sh/python-build-standalone/releases
# Or verify: ./verify-python-version.sh 3.13.9 20251014

RELEASE_DATE="20251014"        # python-build-standalone release date (Oct 14, 2025)
PYTHON_VERSION="3.13.9"        # Python version available in this release
INSTALL_DIR="$(pwd)/.python"
VENV_DIR="$(pwd)/.venv"

# Parse command-line arguments
FORCE_CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force-clean|--clean)
            FORCE_CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "This script always uses portable Python (never system Python)"
            echo ""
            echo "Options:"
            echo "  --force-clean              Clean .python/, .venv/, and uv.lock, then rebuild"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                         # Use existing portable Python or download"
            echo "  $0 --force-clean           # Clean and rebuild everything"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

echo "🐍 Portable Python Setup"
echo "================================"
echo "Always uses self-contained Python"
echo ""

# Handle force clean
if [ "$FORCE_CLEAN" = true ]; then
    echo "🧹 Force clean: Removing all previous installations..."
    rm -rf "$INSTALL_DIR" "$VENV_DIR" uv.lock
    echo "✅ Clean complete"
    echo ""
fi

# Detect OS and Architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "🔍 System Information:"
echo "   OS: $OS"
echo "   Architecture: $ARCH"
echo ""

# Function to download and extract python-build-standalone
download_python_standalone() {
    local version=$1
    local os_type=$2
    local arch_type=$3
    
    # Map OS and architecture to python-build-standalone naming
    case "$os_type" in
        Darwin)
            if [ "$arch_type" = "arm64" ]; then
                platform="aarch64-apple-darwin"
            else
                platform="x86_64-apple-darwin"
            fi
            ;;
        Linux)
            if [ "$arch_type" = "x86_64" ]; then
                platform="x86_64-unknown-linux-gnu"
            elif [ "$arch_type" = "aarch64" ]; then
                platform="aarch64-unknown-linux-gnu"
            fi
            ;;
        *)
            echo "❌ Unsupported OS: $os_type"
            return 1
            ;;
    esac
    
    # Use python-build-standalone releases
    local base_url="https://github.com/astral-sh/python-build-standalone/releases/download"
    local filename="cpython-${version}+${RELEASE_DATE}-${platform}-install_only.tar.gz"
    local download_url="${base_url}/${RELEASE_DATE}/${filename}"
    
    echo "📥 Downloading Python from python-build-standalone..."
    echo "   Version: $version"
    echo "   Release: $RELEASE_DATE"
    echo "   URL: $download_url"
    echo ""
    
    if command -v curl &> /dev/null; then
        curl -L -o "/tmp/${filename}" "$download_url"
    elif command -v wget &> /dev/null; then
        wget -O "/tmp/${filename}" "$download_url"
    else
        echo "❌ Neither curl nor wget found. Please install one."
        return 1
    fi
    
    echo "📦 Extracting Python..."
    mkdir -p "$INSTALL_DIR"
    tar -xzf "/tmp/${filename}" -C "$INSTALL_DIR" --strip-components=1
    rm "/tmp/${filename}"
    
    echo "✅ Python installed to: $INSTALL_DIR"
    return 0
}

# Function to compile Python from source (fallback)
compile_python_from_source() {
    local version=$1
    
    echo "🔨 Compiling Python from source..."
    echo "   This may take 5-15 minutes..."
    
    # Download Python source
    local python_tar="Python-${version}.tar.xz"
    local download_url="https://www.python.org/ftp/python/${version}/${python_tar}"
    
    echo "📥 Downloading Python source..."
    curl -L -o "/tmp/${python_tar}" "$download_url"
    
    # Extract
    echo "📦 Extracting source..."
    tar -xf "/tmp/${python_tar}" -C /tmp
    cd "/tmp/Python-${version}"
    
    # Configure and compile
    echo "⚙️  Configuring..."
    ./configure --prefix="$INSTALL_DIR" --enable-optimizations --with-lto
    
    echo "🔨 Compiling (this takes a while)..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    
    echo "📦 Installing..."
    make install
    
    # Cleanup
    cd - > /dev/null
    rm -rf "/tmp/Python-${version}" "/tmp/${python_tar}"
    
    echo "✅ Python compiled and installed to: $INSTALL_DIR"
}

# Check if portable Python already exists
if [ -f "$INSTALL_DIR/bin/python3" ]; then
    echo "✅ Found existing portable Python: $INSTALL_DIR/bin/python3"
    "$INSTALL_DIR/bin/python3" --version
    PYTHON_CMD="$INSTALL_DIR/bin/python3"
else
    echo "📥 Portable Python not found, will download/compile..."
    echo ""
    
    # Try python-build-standalone first (faster), fallback to source compilation
    if download_python_standalone "$PYTHON_VERSION" "$OS" "$ARCH"; then
        echo "✅ Using pre-built Python"
    else
        echo "⚠️  Pre-built Python not available, compiling from source..."
        compile_python_from_source "$PYTHON_VERSION"
    fi
    
    # Verify Python installation
    PYTHON_CMD="$INSTALL_DIR/bin/python3"
    if [ ! -f "$PYTHON_CMD" ]; then
        # Try alternative path
        PYTHON_CMD="$INSTALL_DIR/python/install/bin/python3"
    fi
    
    if [ ! -f "$PYTHON_CMD" ]; then
        echo "❌ Python binary not found after installation"
        exit 1
    fi
    
    echo ""
    echo "🎉 Python installed successfully!"
    "$PYTHON_CMD" --version
fi

# Clean up venv and lock (always)
echo ""
echo "🧹 Cleaning up virtual environment and lock files..."
rm -rf "$VENV_DIR" uv.lock

# Create virtual environment
echo ""
echo "🔹 Creating virtual environment..."
"$PYTHON_CMD" -m venv "$VENV_DIR"

# Activate and install UV
echo "📦 Installing UV..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install uv --quiet

# Use UV to install dependencies
echo "📦 Installing project dependencies with UV..."
uv lock
uv sync --all-extras

echo ""
echo "✅ Setup complete!"
echo ""
echo "📁 Structure:"
echo "   .python/       - Portable Python $PYTHON_VERSION"
echo "   .venv/         - Virtual environment"
echo "   uv.lock        - Locked dependencies"
echo ""
echo "🎯 Next steps:"
echo ""
echo "   # Activate environment:"
echo "   source .venv/bin/activate"
echo ""
echo "   # Run your application:"
echo "   python src/main.py"
echo ""
echo "   # Run tests with coverage:"
echo "   pytest --cov=src"
echo ""
echo "   # Run tests with coverage report:"
echo "   pytest --cov=src --cov-report=html"
echo ""