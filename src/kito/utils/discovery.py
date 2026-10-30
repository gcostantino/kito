import hashlib
import importlib.util
import sys
from pathlib import Path

_loaded = set()


def load_python_files(directory, strict: bool = True) -> int:
    """Import every public .py file in `directory` (decorators register classes). Returns #files loaded."""
    d = Path(directory).expanduser().resolve()
    if not d.is_dir():
        raise FileNotFoundError(f"Discovery directory not found: {d}")

    tag = hashlib.md5(str(d).encode()).hexdigest()[:8]
    count = 0
    for py in sorted(d.glob('*.py')):
        key = (str(d), py.name)
        if py.name.startswith('_') or key in _loaded:
            continue
        mod_name = f'_kito_user_{tag}.{py.stem}'  # never collides with real packages
        spec = importlib.util.spec_from_file_location(mod_name, py)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)  # don't leave half-loaded modules
            if strict:
                raise
            continue
        _loaded.add(key)
        count += 1
    return count
