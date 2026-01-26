# Contributing to Kito

Thank you for your interest in contributing to Kito! 🎉

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- ✨ Submit pull requests
- 🧪 Write tests
- 💬 Help others in discussions

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/kito.git
cd kito
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev,tensorboard]"
```

### 3. Run Tests

```bash
pytest tests/ -v
```

### 4. Check Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clear, concise code
- Add tests for new features
- Update documentation if needed
- Follow existing code style

### 3. Run Tests

```bash
pytest tests/ -v --cov=kito
```

Ensure:
- All tests pass ✅
- Coverage doesn't decrease

### 4. Commit

```bash
git add .
git commit -m "Add feature: your feature description"
```

Use clear commit messages:
- `feat: add new callback system`
- `fix: resolve DDP initialization bug`
- `docs: update README with examples`
- `test: add tests for preprocessing`

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Code Style

### Python Style

- Use **Black** for formatting (line length: 100)
- Use **type hints** where possible
- Write **docstrings** for public functions/classes
- Keep functions focused and small

### Example

```python
def process_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process input data.

    Args:
        data: Input data array
        normalize: Whether to normalize data

    Returns:
        Processed data array
    """
    if normalize:
        data = (data - data.mean()) / data.std()
    return data
```

## Testing Guidelines

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use descriptive test names
- Test edge cases

### Example Test

```python
def test_normalize_with_constant_data():
    """Test normalization with constant data (edge case)."""
    data = np.ones(10)
    labels = np.zeros(10)

    norm = Normalize()
    data_norm, _ = norm(data, labels)

    # Should remain unchanged (no division by zero)
    assert np.array_equal(data, data_norm)
```

### Running Specific Tests

```bash
# Run single file
pytest tests/unit/test_preprocessing.py -v

# Run single test
pytest tests/unit/test_preprocessing.py::TestNormalize::test_normalize_default_range -v

# Run with markers
pytest -m integration  # Only integration tests
```

## Adding Features

### New Callback

1. Create file in `src/kito/callbacks/`
2. Inherit from `Callback` base class
3. Implement required hooks
4. Add tests in `tests/unit/test_callbacks.py`
5. Update documentation

Example:
```python
from kito.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs, **kwargs):
        print(f"Epoch {epoch} ended")
```

### New Dataset Type

1. Create class in `src/kito/data/datasets.py`
2. Inherit from `KitoDataset`
3. Register with `@DATASETS.register('name')`
4. Implement `_load_sample()` and `__len__()`
5. Add tests
6. Update documentation

### New Preprocessing

1. Create class in `src/kito/data/preprocessing.py`
2. Inherit from `Preprocessing`
3. Register with `@PREPROCESSING.register('name')`
4. Implement `__call__()`
5. Add tests
6. Update documentation

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train(self, epochs: int, callbacks: List[Callback] = None) -> Dict:
    """
    Train the model.

    Args:
        epochs: Number of training epochs
        callbacks: List of callback objects

    Returns:
        Dictionary with training history

    Raises:
        ValueError: If epochs < 1

    Example:
        >>> model.train(epochs=10)
        {'loss': [0.5, 0.4, ...]}
    """
```

### README Updates

When adding features, update:
- Feature list
- Quick start examples
- API reference

## Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try latest version
3. Create minimal reproducible example

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug.

**To Reproduce**
```python
# Minimal code to reproduce
from kito import Engine
...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- Kito version: 0.1.0
- Python version: 3.11
- PyTorch version: 2.0.0
- OS: Ubuntu 22.04

**Additional context**
Any other relevant information.
```

## Feature Requests

Use this template:

```markdown
**Feature Description**
Clear description of the feature.

**Motivation**
Why is this needed? What problem does it solve?

**Proposed Solution**
How would you implement this?

**Alternatives**
Other approaches considered.

**Additional Context**
Examples, mockups, etc.
```

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Give constructive feedback
- Celebrate contributions
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

- 💬 Ask in [GitHub Discussions](https://github.com/yourusername/kito/discussions)
- 🐛 Report bugs in [Issues](https://github.com/yourusername/kito/issues)
- 📧 Email: your.email@example.com

## Thank You! 🙏

Every contribution helps make Kito better. We appreciate your time and effort!
