import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("plotat", os.path.join(HERE, "plot-accesstier.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["plotat"] = mod
spec.loader.exec_module(mod)

mod.fig_deadline_acc()
mod.fig_deadline_lat()
print("regenerated deadline figures", flush=True)
