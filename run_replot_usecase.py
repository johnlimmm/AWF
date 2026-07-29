"""Replot the use-case figure from the saved CSV with a shared top legend
(no overlap with the low bars)."""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "accesstier")
df = pd.read_csv(os.path.join(OUT, "table-usecase.csv"))

PERSONA_ORDER = ["Co-located clinic", "Metro edge", "Wide-area mobile",
                 "Non-terrestrial field", "Heterogeneous"]
METHODS = {"awf": "AWF", "logpool": "Full-Wait (log)",
           "anytime_cbci": "Confidence-Stop", "deadline_aware": "Arrival-Only",
           "generic_voi": "Entropy-Gain"}
CLR = {"awf": "#d1495b", "logpool": "#1b998b", "anytime_cbci": "#2e86ab",
       "deadline_aware": "#5c4d7d", "generic_voi": "#c9a227"}
show = list(METHODS.keys())

pa = (df.groupby(["persona", "method"])["accuracy"].mean().unstack() * 100)
pl = df.groupby(["persona", "method"])["latency"].mean().unstack()

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(15, 6.0))
x = np.arange(len(PERSONA_ORDER))
w = 0.16
bars = []
for i, m in enumerate(show):
    b = ax[0].bar(x + (i - 2) * w, [pa.loc[p, m] for p in PERSONA_ORDER], w,
                  color=CLR[m], label=METHODS[m])
    bars.append(b)
ax[0].set_xticks(x)
ax[0].set_xticklabels(PERSONA_ORDER, rotation=18, ha="right", fontsize=12)
ax[0].set_ylabel("accuracy (%)", fontsize=16)
ax[0].set_ylim(55, 100)
ax[0].set_title("(a) Accuracy across deployment use cases", fontsize=16)
ax[0].tick_params(axis="y", labelsize=13)

for i, m in enumerate(show):
    ax[1].bar(x + (i - 2) * w, [pl.loc[p, m] for p in PERSONA_ORDER], w,
              color=CLR[m])
ax[1].axhline(50, color="0.4", ls=":", lw=1.5)
ax[1].text(len(PERSONA_ORDER) - 0.5, 51, "deadline", fontsize=11,
           color="0.3", ha="right")
ax[1].set_xticks(x)
ax[1].set_xticklabels(PERSONA_ORDER, rotation=18, ha="right", fontsize=12)
ax[1].set_ylabel("mean decision latency (ms)", fontsize=16)
ax[1].set_ylim(0, 56)
ax[1].set_title("(b) Decision latency across use cases", fontsize=16)
ax[1].tick_params(axis="y", labelsize=13)

# single shared legend above both panels (no overlap with bars)
fig.legend([b[0] for b in bars], [METHODS[m] for m in show],
           loc="upper center", ncol=5, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=[0, 0, 1, 0.99])
p = os.path.join(OUT, "fig-usecase.png")
fig.savefig(p)
plt.close(fig)
print("wrote", p)
