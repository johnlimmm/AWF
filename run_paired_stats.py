"""Paired significance test: AWF vs Full-wait log-pool accuracy, per seed
(3-dataset mean), at D=50 and D=100. Reports paired mean difference, 95% CI,
and Wilcoxon signed-rank p-value."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

lines = []
for D in (50, 100):
    d = pd.read_csv(os.path.join(OUT, "table-main-d%d.csv" % D))
    aw = d[d.method == "awf"].groupby("seed")["accuracy"].mean().values * 100
    fw = d[d.method == "logpool"].groupby("seed")["accuracy"].mean().values * 100
    diff = fw - aw  # full-wait minus AWF
    n = len(diff)
    md = diff.mean()
    sd = diff.std(ddof=1)
    sem = sd / np.sqrt(n)
    ci = 1.96 * sem
    line = ("D=%d  Full-wait - AWF: mean diff = %.3f pp, 95%% CI = [%.3f, %.3f]"
            % (D, md, md - ci, md + ci))
    if HAVE_SCIPY:
        try:
            w, p = stats.wilcoxon(fw, aw)
            line += ", Wilcoxon p = %.3f" % p
        except Exception as e:
            line += ", Wilcoxon failed (%s)" % e
        t, pt = stats.ttest_rel(fw, aw)
        line += ", paired t p = %.3f" % pt
    lines.append(line)

txt = "\n".join(lines)
with open(os.path.join(OUT, "paired_stats.txt"), "w") as f:
    f.write(("scipy=%s\n" % HAVE_SCIPY) + txt + "\n")
print("scipy=%s" % HAVE_SCIPY)
print(txt)
