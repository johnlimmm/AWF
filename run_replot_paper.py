import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("plotat", os.path.join(HERE, "plot-accesstier.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["plotat"] = mod
spec.loader.exec_module(mod)

for fn in ("fig_deadline_acc", "fig_deadline_lat", "fig_alpha", "fig_lambda"):
    if hasattr(mod, fn):
        getattr(mod, fn)()
        print("done", fn, flush=True)
    else:
        print("MISSING", fn, flush=True)
print("all paper figures regenerated", flush=True)
