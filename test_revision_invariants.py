"""Small deterministic checks for the IoTJ revision experiment definitions.

These tests target policy endpoints and ablation wiring rather than stochastic
benchmark scores.  They are intentionally independent of MOABB and the raw
network datasets, so they can run before a long experiment campaign.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fusion = _load("revision_fusion", "bcinet-fusion.py")
data = _load("revision_data", "bcinet-data.py")
NetworkProfile = fusion.NetworkProfile


def _profile(delay=(2.0, 20.0)):
    return NetworkProfile(tier="test", delay_range=delay,
                          jitter_range=(0.0, 0.0), loss_prob=0.0)


class RevisionInvariantTests(unittest.TestCase):
    def test_temperature_scaling_preserves_predicted_class(self):
        p = np.array([[0.8, 0.15, 0.05], [0.2, 0.3, 0.5]])
        scaled = data.temperature_scale(p, 2.0)
        np.testing.assert_array_equal(p.argmax(axis=1), scaled.argmax(axis=1))
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)

    def test_prefix_sum_residual_delay_matches_brute_force(self):
        samples = np.array([60.0, 1.0, 15.0, 5.0])  # deliberately unsorted
        profile = NetworkProfile(
            tier="trace", delay_range=(1.0, 60.0),
            jitter_range=(0.0, 0.0), loss_prob=0.2,
            est_samples=samples, est_loss=0.2)
        t, deadline = 10.0, 50.0
        surviving = np.sort(samples)[np.sort(samples) > t]
        remaining = deadline - t
        p_link = 0.8
        denom = 0.2 + p_link * len(surviving) / len(samples)
        brute = (p_link * len(surviving) / len(samples)
                 * np.mean(np.minimum(surviving - t, remaining))
                 + 0.2 * remaining) / denom
        fast = fusion.expected_remaining_delay(profile, t, deadline)
        self.assertAlmostEqual(fast, brute, places=12)

    def test_arrival_only_tau_zero_ignores_impossible_arrival(self):
        P = np.array([[0.7, 0.3], [0.4, 0.6]])
        result = fusion.deadline_aware_waiting(
            P=P, arrivals_ms=np.array([1.0, np.inf]), deadline_ms=50.0,
            profiles=[_profile(), _profile((100.0, 200.0))], tau=0.0)
        self.assertEqual(result.decision_time_ms, 1.0)

    def test_no_margin_ablation_disables_early_margin_stop(self):
        P = np.array([[0.999, 0.001], [0.8, 0.2]])
        kwargs = dict(
            P=P, arrivals_ms=np.array([1.0, 10.0]), deadline_ms=50.0,
            profiles=[_profile(), _profile()],
            confusions=[np.eye(2), np.eye(2)],
            val_acc=np.array([1.0, 1.0]), alpha=0.5,
            lambda_delay=0.0)
        with_margin = fusion.awf_fusion(**kwargs)
        without_margin = fusion.awf_fusion(**kwargs, use_margin=False)
        self.assertEqual(with_margin.stopping_reason, "margin")
        self.assertEqual(with_margin.decision_time_ms, 1.0)
        self.assertEqual(without_margin.decision_time_ms, 10.0)

    def test_accuracy_only_reliability_ablation_is_runnable(self):
        fn = fusion.get_method("awf_accuracy_only")
        result = fn(
            P=np.array([[0.6, 0.4], [0.45, 0.55]]),
            arrivals_ms=np.array([1.0, 10.0]), deadline_ms=50.0,
            profiles=[_profile(), _profile()],
            confusions=[np.eye(2), np.eye(2)],
            val_acc=np.array([0.9, 0.6]), alpha=1e-3,
            lambda_delay=1e-3)
        self.assertIn(result.pred, (0, 1))

    def test_first_k_full_committee_matches_deadline_partial_prediction(self):
        P = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        arrivals = np.array([2.0, 4.0, 7.0])
        first_all = fusion.earliest_k_fusion(P, arrivals, 50.0, k=3)
        at_deadline = fusion.deadline_partial_fusion(P, arrivals, 50.0)
        self.assertEqual(first_all.pred, at_deadline.pred)
        self.assertEqual(first_all.decision_time_ms, 7.0)
        self.assertEqual(at_deadline.decision_time_ms, 50.0)

    def test_confidence_threshold_one_reaches_deadline(self):
        result = fusion.anytime_cbci(
            P=np.array([[0.7, 0.3], [0.6, 0.4]]),
            arrivals_ms=np.array([2.0, 7.0]), deadline_ms=50.0,
            theta=1.0)
        self.assertEqual(result.decision_time_ms, 50.0)
        self.assertEqual(result.stopping_reason, "deadline")

    def test_dp_reference_is_non_clairvoyant_and_cost_sensitive(self):
        kwargs = dict(
            P=np.array([[0.6, 0.4], [0.9, 0.1]]),
            arrivals_ms=np.array([1.0, 10.0]), deadline_ms=20.0,
            profiles=[_profile(), _profile((2.0, 12.0))],
            confusions=[np.eye(2), np.eye(2)], grid_ms=5.0)
        wait_result = fusion.dp_optimal_stopping(**kwargs, lambda_delay=0.0)
        stop_result = fusion.dp_optimal_stopping(**kwargs, lambda_delay=1.0)
        self.assertEqual(wait_result.decision_time_ms, 10.0)
        self.assertEqual(stop_result.decision_time_ms, 1.0)
        self.assertEqual(stop_result.stopping_reason, "dp_stop")


if __name__ == "__main__":
    unittest.main()
