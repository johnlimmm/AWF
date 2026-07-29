import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tb", os.path.join(HERE, "bcinet-tracebank.py"))
tb = importlib.util.module_from_spec(spec)
sys.modules["tb"] = tb
spec.loader.exec_module(tb)

import numpy as np
bank = tb.build_tracebank(seed=0, verbose=False)
print("Rebuilt tracebank. Per-tier one-way delay percentiles + loss:")
for t in tb.TIER_ORDER:
    b = bank[t]
    d = b.all_delays()
    print("%-4s p50=%.1f p90=%.1f p95=%.1f p99=%.1f  loss=%.4f  n=%d"
          % (t, np.percentile(d, 50), np.percentile(d, 90), np.percentile(d, 95),
             np.percentile(d, 99), b.overall_loss(), d.size))
