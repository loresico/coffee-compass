#!/usr/bin/env bash
# Script to verify if a Python version exists in a specific release

PYTHON_VERSION="${1:-3.12.7}"
RELEASE_DATE="${2:-20250918}"

echo "🔍 Checking Python $PYTHON_VERSION in release $RELEASE_DATE"
echo ""

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

case "$OS" in
    Darwin)
        if [ "$ARCH" = "arm64" ]; then
            PLATFORM="aarch64-apple-darwin"
        else
            PLATFORM="x86_64-apple-darwin"
        fi
        ;;
    Linux)
        if [ "$ARCH" = "x86_64" ]; then
            PLATFORM="x86_64-unknown-linux-gnu"
        elif [ "$ARCH" = "aarch64" ]; then
            PLATFORM="aarch64-unknown-linux-gnu"
        fi
        ;;
esac

echo "Platform: $PLATFORM"
echo ""

# Construct URL
BASE_URL="https://github.com/astral-sh/python-build-standalone/releases/download"
FILENAME="cpython-${PYTHON_VERSION}+${RELEASE_DATE}-${PLATFORM}-install_only.tar.gz"
URL="${BASE_URL}/${RELEASE_DATE}/${FILENAME}"

echo "Testing URL:"
echo "$URL"
echo ""

# Check if URL exists
if curl --output /dev/null --silent --head --fail "$URL"; then
    echo "✅ SUCCESS: Python $PYTHON_VERSION exists in release $RELEASE_DATE"
    echo ""
    echo "You can use:"
    echo "  PYTHON_VERSION=\"$PYTHON_VERSION\""
    echo "  release_date=\"$RELEASE_DATE\""
    exit 0
else
    echo "❌ NOT FOUND: Python $PYTHON_VERSION does not exist in release $RELEASE_DATE"
    echo ""
    echo "💡 Suggestions:"
    echo "  1. Check available versions at:"
    echo "     https://github.com/astral-sh/python-build-standalone/releases/tag/$RELEASE_DATE"
    echo ""
    echo "  2. Try latest release (20250918) with latest Python patches:"
    echo "     Python 3.14.x, 3.13.x, 3.12.x, 3.11.x, 3.10.x"
    echo ""
    echo "  3. Use this script to test:"
    echo "     ./verify-python-version.sh 3.13.9 20250918"
    exit 1
fi
