import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "results", "accesstier", "table-deadline.csv"))
t = d.groupby(["method", "deadline_ms"])["accuracy"].mean().unstack("method") * 100
lat = d.groupby(["method", "deadline_ms"])["decision_latency"].mean().unstack("method")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
out = []
out.append("ACCURACY (%)")
out.append(t.round(2).to_string())
out.append("")
out.append("LATENCY (ms)")
out.append(lat.round(2).to_string())
txt = "\n".join(out)
with open(os.path.join(HERE, "results", "accesstier", "deadline_inspect.txt"), "w") as f:
    f.write(txt + "\n")
print(txt)
