# Contributing to FakeLenseV2

First off, thank you for considering contributing to FakeLenseV2! It's people like you that make this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by a simple principle: **Be respectful and constructive**.

By participating, you are expected to:
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and PyTorch
- Familiarity with reinforcement learning (helpful but not required)

### Development Setup

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/FakeLenseV2.git
   cd FakeLenseV2
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/Navy10021/FakeLenseV2.git
   ```

4. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install in development mode**

   ```bash
   pip install -e ".[dev]"
   ```

   This installs:
   - All project dependencies
   - Development tools (pytest, black, flake8, mypy)
   - The package in editable mode

6. **Verify installation**

   ```bash
   pytest tests/
   python -m code.main --help
   ```

---

## 🔄 Development Workflow

### Staying Up-to-Date

Always sync with the upstream repository before starting new work:

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

### Creating a Branch

Create a descriptive branch name:

```bash
# For features
git checkout -b feature/add-multilingual-support

# For bug fixes
git checkout -b fix/model-loading-error

# For documentation
git checkout -b docs/update-api-guide
```

### Making Changes

1. **Make your changes**
   - Write clean, readable code
   - Follow the project structure
   - Add docstrings to functions
   - Update relevant documentation

2. **Test your changes**
   ```bash
   pytest tests/ -v
   ```

3. **Format your code**
   ```bash
   black code/ tests/
   flake8 code/ tests/
   ```

4. **Type check (optional but recommended)**
   ```bash
   mypy code/ --ignore-missing-imports
   ```

---

## 📐 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use absolute imports from project root
- **Quotes**: Prefer double quotes for strings

### Code Formatting

We use **Black** for automatic code formatting:

```bash
black code/ tests/
```

### Type Hints

Add type hints to function signatures:

```python
def predict(
    text: str,
    source: str,
    social_reactions: float
) -> int:
    """
    Predict news authenticity.

    Args:
        text: Article text content
        source: News source name
        social_reactions: Number of social media reactions

    Returns:
        Predicted class (0=Fake, 1=Suspicious, 2=Real)
    """
    pass
```

### Documentation

All public functions must have docstrings:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of what the function does.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Examples:
        >>> function_name(value1, value2)
        expected_output
    """
    pass
```

### Project Structure

When adding new files, follow the existing structure:

```
code/
├── models/          # Neural network architectures
├── agents/          # RL agents
├── utils/           # Utility functions
├── train.py         # Training scripts
├── inference.py     # Inference logic
├── main.py          # CLI interface
└── api_server.py    # API server
```

---

## 🧪 Testing

### Writing Tests

Add tests for new features:

```python
# tests/test_my_feature.py
import pytest
from code.my_module import my_function


class TestMyFeature:
    """Tests for my new feature"""

    def test_basic_functionality(self):
        """Test basic use case"""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_cases(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    def test_with_mock_data(self):
        """Test with mocked dependencies"""
        # Test implementation
        pass
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run tests matching a pattern
pytest tests/ -k "test_inference"
```

### Test Coverage

Aim for at least 80% coverage for new code:

```bash
pytest tests/ --cov=code --cov-report=term-missing
```

---

## 📤 Submitting Changes

### Before Submitting

1. **Ensure all tests pass**
   ```bash
   pytest tests/ -v
   ```

2. **Format code**
   ```bash
   black code/ tests/
   ```

3. **Check code quality**
   ```bash
   flake8 code/ tests/
   ```

4. **Update documentation**
   - Update docstrings
   - Update README if needed
   - Add examples if applicable

### Commit Messages

Write clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat: Add multi-language support for text vectorization

- Add language detection using langdetect
- Support for Korean, Spanish, and French
- Update vectorizer to handle multilingual input
- Add tests for new functionality

Closes #42
```

```
fix: Resolve model loading error on CPU-only machines

The model was trying to load CUDA tensors on CPU-only machines.
Added proper device mapping in the load function.

Fixes #38
```

### Creating a Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**
   - Go to your fork on GitHub
   - Click "Pull Request"
   - Fill out the PR template
   - Link related issues

3. **PR Title Format**
   ```
   [Type] Brief description
   ```
   Example: `[Feature] Add multilingual support`

4. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Motivation
   Why is this change needed?

   ## Changes Made
   - Change 1
   - Change 2
   - Change 3

   ## Testing
   How was this tested?

   ## Checklist
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] Code formatted with Black
   - [ ] All tests passing
   - [ ] No breaking changes (or noted in description)
   ```

---

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the bug is already fixed
3. **Collect relevant information**

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- FakeLenseV2 version: [e.g., 2.0.0]
- PyTorch version: [e.g., 1.12.0]

**Additional Context**
- Error messages
- Stack traces
- Screenshots
- Relevant code snippets
```

---

## 💡 Suggesting Enhancements

We welcome suggestions for new features!

### Enhancement Proposal Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature useful?

**Proposed Implementation**
How could this be implemented?

**Alternatives Considered**
What other approaches did you consider?

**Additional Context**
Mockups, examples, or references
```

---

## 📚 Areas for Contribution

Looking for ideas? Here are some areas where we need help:

### High Priority

- 🌐 **Multi-language support** - Extend to Korean, Spanish, French
- 📊 **Explainable AI** - Add attention visualization
- 🧪 **More tests** - Increase coverage to 90%+
- 📖 **Documentation** - More examples and tutorials

### Medium Priority

- 🔧 **Performance optimization** - Model quantization, caching
- 🎨 **UI/Dashboard** - Real-time monitoring interface
- 📱 **Mobile deployment** - ONNX export, TensorFlow Lite
- 🔌 **Integrations** - Twitter API, Facebook API

### Good First Issues

Issues labeled `good-first-issue` are great for beginners:
- Add more unit tests
- Improve docstrings
- Fix typos in documentation
- Add code examples

---

## 🎯 Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow project conventions?
- **Impact**: Does it improve the project?

### Timeline

- Initial review: Within 3-5 days
- Follow-up: Within 1-2 days
- Merge: After approval and CI passes

### After Your PR is Merged

1. **Delete your branch** (optional)
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Sync with upstream**
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

3. **Celebrate!** 🎉 You're now a FakeLenseV2 contributor!

---

## 🙏 Recognition

Contributors will be:
- Listed in the project README
- Credited in release notes
- Acknowledged in academic publications (for significant contributions)

---

## 📞 Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bug reports and feature requests
- **Email**: iyunseob4@gmail.com for private inquiries

---

## 📖 Additional Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

Thank you for contributing to FakeLenseV2! Every contribution, no matter how small, makes a difference. 🚀
