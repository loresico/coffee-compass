"""
Tests for main module.

Run with: pytest
Run with coverage: pytest --cov=src
"""
import pytest
from src.main import greet


class TestGreet:
    """Tests for the greet function."""
    
    def test_greet_default(self):
        """Test greet with default argument."""
        result = greet()
        assert result == "Hello, World!"
    
    def test_greet_with_name(self):
        """Test greet with custom name."""
        result = greet("Python")
        assert result == "Hello, Python!"
    
    def test_greet_empty_string(self):
        """Test greet with empty string."""
        result = greet("")
        assert result == "Hello, !"
    
    @pytest.mark.parametrize("name,expected", [
        ("Alice", "Hello, Alice!"),
        ("Bob", "Hello, Bob!"),
        ("123", "Hello, 123!"),
        ("🐍", "Hello, 🐍!"),
    ])
    def test_greet_parametrized(self, name, expected):
        """Test greet with multiple inputs using parametrize."""
        assert greet(name) == expected


class TestMain:
    """Tests for the main function."""
    
    def test_main_runs_without_error(self, capsys):
        """Test that main function runs without raising exceptions."""
        from src.main import main
        
        main()
        
        # Capture output
        captured = capsys.readouterr()
        
        # Check that something was printed
        assert "Your UV project is running" in captured.out
        assert "Hello, World!" in captured.out