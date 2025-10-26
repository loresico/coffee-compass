"""
Main application entry point.

This is a simple example to get you started.
Replace this with your actual application code.
"""


def greet(name: str = "World") -> str:
    """
    Return a greeting message.
    
    Args:
        name: Name to greet
        
    Returns:
        Greeting message
        
    Examples:
        >>> greet()
        'Hello, World!'
        >>> greet("Python")
        'Hello, Python!'
    """
    return f"Hello, {name}!"


def main() -> None:
    """Main function - application entry point."""
    print("🚀 Your UV project is running!")
    print(greet())
    print("\n✨ Edit src/main.py to build your application")
    print("📝 Add your dependencies to pyproject.toml")
    print("🧪 Run tests with: pytest")


if __name__ == "__main__":
    main()