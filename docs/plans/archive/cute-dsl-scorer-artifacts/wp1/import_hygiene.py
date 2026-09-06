import importlib, importlib.abc, sys
m = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
print("cutlass imported at module import:", "cutlass" in sys.modules)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] == "cutlass":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
sys.meta_path.insert(0, Block())
try:
    m.ensure_built()
    print("ensure_built: unexpectedly succeeded")
except m.CuteMissing as e:
    print("ensure_built without cutlass -> CuteMissing:", str(e)[:70])
except ImportError as e:
    print("ensure_built without cutlass -> plain ImportError (WRONG):", type(e).__name__, e)
print("is_available():", m.is_available())
