# Contributing to Kito

Bug reports, fixes, documentation improvements and new features are welcome. For anything larger than a small
fix, open an issue first so the design can be discussed before you write code.

## Setup

```bash
git clone [https://github.com/](https://github.com/)<your-username>/kito.git
cd kito
pip install -e ".[dev]"
```

Add the `docs` extra if you also build the documentation: `pip install -e ".[dev,docs]"`.

## Checks before opening a pull request

```bash
ruff check src/
pytest tests/unit -v
sphinx-build -W --keep-going -b html docs/source docs/_build/html
```

- Lint: CI runs `ruff check src/`.
- Tests: the suite is being brought back in line with the current API and CI does not run it yet. Run at
  least the tests for the code you changed, and add tests for new behavior. Unit tests live in `tests/unit/`,
  integration tests in `tests/integration/`. Tests marked with markers must use markers declared in
  `pytest.ini`, because `--strict-markers` is on.
- Docs: the build treats warnings as errors, both locally and on Read the Docs.

## Pull requests

1. Create a branch from `main`.
2. Keep the change focused; unrelated cleanups go in a separate pull request.
3. Update the documentation and `CHANGELOG.md` when behavior changes.
4. In the description, explain what changed and why, and link the related issue.

Commit messages are short and in the imperative. A prefix such as `fix:`, `feat:`, `docs:` or `test:` is
welcome but not required.

## Code style

- Follow the style of the surrounding code and add type hints to public functions.
- Public classes and functions need Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
- Code samples in docstrings must be introduced with `::` followed by a blank line, otherwise Sphinx
  reports a warning:

```python
def scale(x, factor=2.0):
    """
    Multiply ``x`` by ``factor``.

    Args:
        x: Input array.
        factor: Multiplier.

    Example::

        y = scale(x, factor=3.0)
    """
    return x * factor
```

## Adding components

### Callback

Subclass `Callback` in `src/kito/callbacks/`, export it from `src/kito/callbacks/__init__.py`, and add tests
in `tests/unit/test_callbacks.py`. Hooks receive keyword arguments, so accept `**kwargs`:

```python
from kito.callbacks import Callback


class PrintLoss(Callback):
    def on_epoch_end(self, epoch, engine, model, logs=None, **kwargs):
        print(f"epoch {epoch}: {logs}")
```

### Dataset

Subclass `KitoDataset` (`src/kito/data/datasets.py`), implement `_load_sample` and `__len__`, and register it
with `@DATASETS.register('name')` so it can be selected with `config.data.dataset_type`.

### Preprocessing step

Subclass `Preprocessing` (`src/kito/data/preprocessing.py`), implement its abstract methods `__call__` and
`_transform`, and register it with `@PREPROCESSING.register('name')`. Steps must take statistics as explicit
parameters rather than estimating them from the sample. Add tests in `tests/unit/test_preprocessing.py`.

## Reporting bugs

Include:

- A minimal script that reproduces the problem.
- The expected and the actual behavior, with the full traceback.
- Kito version (`python -c "import kito; print(kito.__version__)"`), Python version, PyTorch version, OS, and
  whether you ran on CPU, one GPU or DDP.

## Feature requests

Describe the problem you want to solve before the proposed API, and say what you currently do instead.

## License

By contributing you agree that your contributions are licensed under the project's MIT license.
