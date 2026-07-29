"""
bcinet-data.py
==============

Data loading and per-user (per-subject) classifier training for the
distributed collaborative BCI (cBCI) experimental pipeline.

Each MOABB subject is treated as one *user* of the collaborative BCI.
For every user we produce:

    * test posterior probability vectors (one per test trial)
    * test true labels
    * validation accuracy and validation macro F1 (used as user reliability)
    * confusion matrix P_i(z_i | c) estimated on the validation set
      (rows are true class c, columns are predicted class z_i, row-normalized)

These outputs are the inputs for the fusion / waiting-policy experiments
implemented in ``bcinet-fusion.py``.

The module is intentionally self-contained: the heavy work happens in
:func:`build_user_pool`, which returns a list of :class:`UserInfo` objects.
We cache the result on disk so that subsequent runs of the experiment do
not have to re-download or re-train.

Notes
-----
* MI (motor imagery) datasets use a band-pass filter 8-30 Hz, CSP with 6
  components, and a Shrinkage LDA classifier.
* P300 datasets flatten each epoch to a (channel * time) vector, apply
  StandardScaler, and use a Shrinkage LDA classifier.
* All classifiers expose ``predict_proba`` so that the fusion layer can
  combine probabilistic outputs.
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Paradigm of every supported dataset. ``mi`` = motor imagery, ``p300`` = P300.
DATASET_PARADIGM: Dict[str, str] = {
    "BNCI2014_001": "mi",
    "Zhou2016":     "mi",
    "Weibo2014":    "mi",
}

SUPPORTED_DATASETS = list(DATASET_PARADIGM.keys())


# MI datasets that should keep ALL of their motor-imagery classes instead of
# the default left/right binary reduction. Maps dataset name -> n_classes.
# The benchmark now uses only multi-class imagery datasets so that the
# margin threshold and per-user risk-reduction terms are non-trivial.
MI_MULTICLASS: Dict[str, int] = {
    "BNCI2014_001": 4,   # left hand, right hand, feet, tongue
    "Zhou2016":     3,   # left hand, right hand, feet
    "Weibo2014":    7,   # L/R hand, hands, feet, two combos, rest
}


# ---------------------------------------------------------------------------
# UserInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserInfo:
    """Per-user (per-subject) bundle used everywhere downstream."""

    user_id: str                    # e.g. "Cho2017_S03"
    dataset: str                    # dataset name
    paradigm: str                   # "mi" or "p300"
    n_classes: int                  # number of target classes

    # Test-time outputs (one entry per test trial)
    test_posteriors: np.ndarray     # shape (n_test, n_classes), rows sum to 1
    test_labels: np.ndarray         # shape (n_test,) integer class ids

    # Validation-set diagnostics
    val_accuracy: float             # validation accuracy
    val_macro_f1: float             # validation macro-averaged F1
    confusion: np.ndarray           # row-normalized P(z=col | c=row), shape (C, C)

    # Network-related fields are filled in later (bcinet-network.py)
    network_tier: Optional[str] = None
    delay_range: Optional[Tuple[float, float]] = None
    jitter_range: Optional[Tuple[float, float]] = None
    loss_prob: Optional[float] = None

    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MOABB loading helpers
# ---------------------------------------------------------------------------

def _import_moabb_components(paradigm: str):
    """Lazy-import MOABB and the appropriate paradigm/dataset classes.

    Returns a tuple ``(dataset_classes_dict, paradigm_object)``. The
    dataset_classes_dict maps the dataset name string to the class that
    instantiates that dataset.
    """
    try:
        import moabb  # noqa: F401
        from moabb.paradigms import LeftRightImagery, MotorImagery, P300
        from moabb import datasets as moabb_datasets
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MOABB is required to load EEG datasets. Install it via\n"
            "    pip install moabb\n"
            f"Original error: {exc}"
        ) from exc

    name_to_cls: Dict[str, type] = {}
    for ds_name in SUPPORTED_DATASETS:
        cls = getattr(moabb_datasets, ds_name, None)
        if cls is not None:
            name_to_cls[ds_name] = cls

    if paradigm == "mi":
        # Many MI datasets have only two classes (left/right hand);
        # LeftRightImagery handles that gracefully. For multi-class MI
        # datasets we fall back to the generic MotorImagery paradigm.
        return name_to_cls, ("LeftRightImagery", LeftRightImagery,
                             "MotorImagery", MotorImagery)
    elif paradigm == "p300":
        return name_to_cls, ("P300", P300)
    else:
        raise ValueError(f"Unknown paradigm {paradigm!r}")


def _instantiate_paradigm(paradigm_name: str, dataset_cls):
    """Pick the right MOABB paradigm object for a given dataset."""
    from moabb.paradigms import LeftRightImagery, MotorImagery, P300

    if paradigm_name == "mi":
        # Try LeftRight first; if the dataset isn't compatible, switch
        # to generic MotorImagery. We use the canonical 8-30 Hz band.
        try:
            return LeftRightImagery(fmin=8.0, fmax=30.0)
        except Exception:
            return MotorImagery(fmin=8.0, fmax=30.0)
    elif paradigm_name == "p300":
        return P300()
    raise ValueError(paradigm_name)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _make_mi_pipeline(n_components: int = 6) -> Pipeline:
    """Build the MI pipeline: CSP(6) -> Shrinkage LDA(predict_proba)."""
    # MNE's CSP is the de-facto standard for MOABB MI pipelines.
    from mne.decoding import CSP

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([("csp", csp), ("lda", lda)])


def _make_p300_pipeline() -> Pipeline:
    """Build the P300 pipeline: Flatten -> StandardScaler -> Shrinkage LDA."""
    from sklearn.base import BaseEstimator, TransformerMixin

    class _Flatten(BaseEstimator, TransformerMixin):
        """Flatten (n_trials, n_channels, n_times) -> (n_trials, F)."""

        def fit(self, X, y=None):  # noqa: D401
            return self

        def transform(self, X):
            X = np.asarray(X)
            return X.reshape(X.shape[0], -1)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    return Pipeline([
        ("flatten", _Flatten()),
        ("scaler", StandardScaler()),
        ("lda", lda),
    ])


# ---------------------------------------------------------------------------
# Per-subject training
# ---------------------------------------------------------------------------

def _train_one_subject(
    X: np.ndarray,
    y: np.ndarray,
    paradigm_name: str,
    rng: np.random.RandomState,
    val_size: float = 0.2,
    test_size: float = 0.3,
    min_trials: int = 30,
) -> Optional[UserInfo]:
    """Train per-subject classifier and produce a :class:`UserInfo` bundle.

    Performs a stratified train/val/test split, fits the appropriate
    pipeline on the training set, computes validation diagnostics, and
    finally returns the test-time posteriors used as user responses.
    """
    # Convert string class labels (e.g. "left_hand") to integer ids ----------
    classes, y_int = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    if n_classes < 2 or len(y_int) < min_trials:
        return None

    # train+val / test split, then val carved out of train -----------------
    try:
        X_tv, X_te, y_tv, y_te = train_test_split(
            X, y_int, test_size=test_size,
            random_state=rng.randint(0, 2**31 - 1), stratify=y_int)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=val_size,
            random_state=rng.randint(0, 2**31 - 1), stratify=y_tv)
    except ValueError:
        # Not enough samples per class for stratified split
        return None

    # Build & fit pipeline ---------------------------------------------------
    if paradigm_name == "mi":
        pipe = _make_mi_pipeline()
    else:
        pipe = _make_p300_pipeline()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
    except Exception as exc:  # pragma: no cover - dataset specific issues
        warnings.warn(f"Pipeline fit failed: {exc}")
        return None

    # Validation diagnostics ------------------------------------------------
    val_pred = pipe.predict(X_va)
    val_acc = float(accuracy_score(y_va, val_pred))
    val_f1 = float(f1_score(y_va, val_pred, average="macro"))

    # Confusion matrix P(z = col | c = row), row-normalized -----------------
    cm = confusion_matrix(y_va, val_pred, labels=np.arange(n_classes)).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    confusion_norm = cm / row_sums
    # Add a tiny smoothing so that no entry is exactly zero. This avoids
    # log(0) when the validation set never observed a (c, z) pair.
    confusion_norm = (confusion_norm + 1e-3)
    confusion_norm = confusion_norm / confusion_norm.sum(axis=1, keepdims=True)

    # Test-time posteriors ---------------------------------------------------
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_te)
    else:  # pragma: no cover
        # Fall back to one-hot of predictions if needed.
        preds = pipe.predict(X_te)
        proba = np.eye(n_classes)[preds]

    # Make sure each row is a valid distribution.
    proba = np.clip(proba, 1e-9, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)

    return UserInfo(
        user_id="",            # filled in by caller
        dataset="",            # filled in by caller
        paradigm=paradigm_name,
        n_classes=n_classes,
        test_posteriors=proba.astype(np.float64),
        test_labels=y_te.astype(np.int64),
        val_accuracy=val_acc,
        val_macro_f1=val_f1,
        confusion=confusion_norm.astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Public API: build / cache user pool
# ---------------------------------------------------------------------------

def build_user_pool(
    dataset_name: str,
    cache_dir: str = "cache",
    seed: int = 0,
    max_subjects: Optional[int] = None,
    force_rebuild: bool = False,
) -> List[UserInfo]:
    """Load a MOABB dataset and produce a user pool.

    Parameters
    ----------
    dataset_name : str
        One of :data:`SUPPORTED_DATASETS`.
    cache_dir : str
        Directory used to cache the trained user pool to disk.
    seed : int
        Base random seed used for the train/val/test splits.
    max_subjects : int, optional
        If given, truncate the subject list (useful for quick runs).
    force_rebuild : bool
        Ignore the cache and rebuild from scratch.

    Returns
    -------
    list of UserInfo
        One :class:`UserInfo` per successfully-trained subject.
    """
    if dataset_name not in DATASET_PARADIGM:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            f"Choose from {SUPPORTED_DATASETS}."
        )

    paradigm_name = DATASET_PARADIGM[dataset_name]

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"users-{dataset_name}-seed{seed}.pkl"
    )
    if (not force_rebuild) and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # ------------------------------------------------------------------
    # MOABB load
    # ------------------------------------------------------------------
    try:
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import LeftRightImagery, MotorImagery, P300
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "MOABB is required. Install with `pip install moabb`. "
            f"Original error: {exc}"
        ) from exc

    cls = getattr(moabb_datasets, dataset_name, None)
    if cls is None:
        raise ValueError(
            f"MOABB does not expose dataset {dataset_name!r}. "
            "Please check the MOABB version (`pip install -U moabb`)."
        )
    dataset = cls()

    # Pick paradigm object (some MI datasets are not LeftRight-compatible)
    if paradigm_name == "mi":
        if dataset_name in MI_MULTICLASS:
            # Keep all motor-imagery classes for this dataset.
            paradigm = MotorImagery(
                fmin=8.0, fmax=30.0, n_classes=MI_MULTICLASS[dataset_name])
        else:
            try:
                paradigm = LeftRightImagery(fmin=8.0, fmax=30.0)
                # Validation: paradigm.datasets must include this one
                X_probe, y_probe, meta_probe = paradigm.get_data(
                    dataset=dataset, subjects=[dataset.subject_list[0]]
                )
                if len(np.unique(y_probe)) < 2:
                    raise ValueError("LeftRightImagery returned <2 classes")
            except Exception:
                paradigm = MotorImagery(fmin=8.0, fmax=30.0)
    else:
        paradigm = P300()

    rng = np.random.RandomState(seed)

    users: List[UserInfo] = []
    subject_list = list(dataset.subject_list)
    if max_subjects is not None:
        subject_list = subject_list[:max_subjects]

    for sid in subject_list:
        try:
            X, y, _ = paradigm.get_data(dataset=dataset, subjects=[sid])
        except Exception as exc:
            warnings.warn(
                f"[{dataset_name}] Failed to download/load subject {sid}: {exc}\n"
                "Hint: this often happens because of a network error or because "
                "the dataset's host requires manual re-download. Try running "
                "`python -c \"from moabb.utils import set_download_dir; set_download_dir('mne_data')\"` "
                "or check your internet connection."
            )
            continue

        ui = _train_one_subject(np.asarray(X), np.asarray(y),
                                paradigm_name=paradigm_name, rng=rng)
        if ui is None:
            continue
        ui.user_id = f"{dataset_name}_S{int(sid):02d}"
        ui.dataset = dataset_name
        users.append(ui)

    if not users:
        raise RuntimeError(
            f"No usable subjects produced for dataset {dataset_name!r}. "
            "Check the MOABB cache and the warnings printed above."
        )

    with open(cache_path, "wb") as fh:
        pickle.dump(users, fh)
    return users


# ---------------------------------------------------------------------------
# Trial sampling helpers
# ---------------------------------------------------------------------------


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (probit) via Acklam's rational approximation.

    Self-contained (no SciPy dependency); absolute error < 1.2e-9 on (0,1).
    Used only by the correlated-error committee sampler.
    """
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def align_test_trials(users: List[UserInfo], rng: np.random.RandomState,
                      n_trials: Optional[int] = None,
                      rho_err: float = 0.0,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Align test trials across users and return ``(P, y)``.

    Different subjects naturally have different numbers of test trials and
    no shared trial identity. For the collaborative BCI experiment we
    *simulate* a multi-user committee evaluating the same trial by
    bootstrap-sampling, for every ``user`` and every trial index ``t``,
    one of that user's test trials whose true label matches the simulated
    ground-truth label.

    The simulated ground-truth labels are drawn (uniformly) from the set
    of class ids that *all* users contain in their test set. This keeps
    every user able to contribute a posterior for every trial.

    Parameters
    ----------
    users : list of UserInfo
    rng : numpy RandomState
    n_trials : int, optional
        Number of simulated trials. Defaults to the smallest user's test
        set size, capped at 400 trials for tractability.
    rho_err : float, optional
        Cross-user error-correlation strength in ``[0, 1)``. ``0`` (default)
        reproduces the conditionally-independent committee (each user
        bootstraps uniformly from its label-matching trials). For
        ``rho_err > 0`` a shared latent *trial hardness* factor is injected
        via a Gaussian copula: a common standard-normal :math:`Z_t` plus an
        idiosyncratic :math:`\\varepsilon_i` decide whether user ``i`` draws
        from its *incorrect* or *correct* label-matching pool, with user
        ``i`` erring iff
        :math:`\\sqrt{\\rho}Z_t+\\sqrt{1-\\rho}\\varepsilon_i<\\Phi^{-1}(e_{i,c})`.
        Because the threshold equals the probit of the user's empirical
        per-class error rate :math:`e_{i,c}`, every user's *marginal*
        accuracy is preserved exactly; only the cross-user error
        *correlation* changes. This isolates the effect of the committee
        independence assumption on fusion gain.

    Returns
    -------
    P : ndarray, shape (n_users, n_trials, n_classes)
        Posterior probability cube.
    y : ndarray, shape (n_trials,)
        Simulated ground-truth labels.
    """
    n_classes = users[0].n_classes
    for u in users:
        if u.n_classes != n_classes:
            raise ValueError(
                "All users must share the same number of classes; got "
                f"{[u.n_classes for u in users]}"
            )

    # Per-user, per-class index lookup -------------------------------------
    per_user_idx_by_cls: List[Dict[int, np.ndarray]] = []
    common_classes: Optional[set] = None
    for u in users:
        d = {c: np.where(u.test_labels == c)[0] for c in range(n_classes)
             if np.any(u.test_labels == c)}
        per_user_idx_by_cls.append(d)
        cls_set = set(d.keys())
        common_classes = cls_set if common_classes is None else (common_classes & cls_set)

    if not common_classes:
        raise RuntimeError(
            "No class is present in every user's test set; cannot align trials."
        )
    common_classes_arr = np.array(sorted(common_classes))

    if n_trials is None:
        n_trials = min(min(len(u.test_labels) for u in users), 400)

    y_true = rng.choice(common_classes_arr, size=n_trials)

    P = np.zeros((len(users), n_trials, n_classes), dtype=np.float64)

    if rho_err <= 0.0:
        # Conditionally-independent committee (original behaviour) ----------
        for ui, u in enumerate(users):
            for t in range(n_trials):
                c = int(y_true[t])
                cand = per_user_idx_by_cls[ui][c]
                j = cand[rng.randint(0, len(cand))]
                P[ui, t, :] = u.test_posteriors[j]
        return P, y_true

    # Correlated-error committee via shared latent trial hardness ----------
    rho_err = float(min(rho_err, 0.999))
    # Split each user's per-class candidate pool into correct / incorrect and
    # precompute the probit threshold of the empirical per-class error rate.
    correct_pool: List[Dict[int, np.ndarray]] = []
    incorrect_pool: List[Dict[int, np.ndarray]] = []
    thresh: List[Dict[int, float]] = []
    for ui, u in enumerate(users):
        cp: Dict[int, np.ndarray] = {}
        ip: Dict[int, np.ndarray] = {}
        th: Dict[int, float] = {}
        preds = np.argmax(u.test_posteriors, axis=1)
        for c, cand in per_user_idx_by_cls[ui].items():
            ok = cand[preds[cand] == c]
            bad = cand[preds[cand] != c]
            cp[c] = ok if len(ok) else cand
            ip[c] = bad if len(bad) else cand
            e_ic = len(bad) / len(cand)
            th[c] = _norm_ppf(min(max(e_ic, 1e-6), 1.0 - 1e-6))
        correct_pool.append(cp)
        incorrect_pool.append(ip)
        thresh.append(th)

    s_shared = np.sqrt(rho_err)
    s_idio = np.sqrt(1.0 - rho_err)
    Z = rng.standard_normal(size=n_trials)            # shared hardness
    for t in range(n_trials):
        c = int(y_true[t])
        zt = Z[t]
        eps = rng.standard_normal(size=len(users))
        latent = s_shared * zt + s_idio * eps
        for ui, u in enumerate(users):
            if latent[ui] < thresh[ui][c]:
                pool = incorrect_pool[ui][c]
            else:
                pool = correct_pool[ui][c]
            j = pool[rng.randint(0, len(pool))]
            P[ui, t, :] = u.test_posteriors[j]
    return P, y_true
