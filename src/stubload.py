"""Load the repository's rule/decision functions without heavy deps.
PyMuPDF, torch, transformers and scikit-learn are only used for PDF parsing,
local generation and TF-IDF retrieval, none of which this analysis calls.
They are replaced by inert stub modules so the pure-Python rule code can be imported."""
import sys, types, importlib.abc, importlib.machinery

STUB_ROOTS = {"fitz", "torch", "transformers", "sklearn", "numpy", "scipy"}

class _Anything:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Anything()
    def __getattr__(self, n): return _Anything()
    def __mro_entries__(self, bases): return (object,)

class _StubModule(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Anything()

class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUB_ROOTS:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None
    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []; return m
    def exec_module(self, module): pass

sys.meta_path.insert(0, _Finder())

def load(repo):
    sys.path.insert(0, f"{repo}/src")
    import ccf_a_upgrade_study as m
    return m
