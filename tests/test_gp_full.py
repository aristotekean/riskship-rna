"""Checks for holdout/group leakage and symbolic model correctness."""
import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gp_full_experiment import (Config, Encoder, evolution, exact_f1_threshold,
                                predict_tree)
from gp_shipment_loss import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def frame():
    data = pd.DataFrame({"SHIPMENT_ID": ["a", "a", "b", "c", "c", "d"],
                         "date_status_0": pd.to_datetime(["2025-09-01"] * 3 + ["2025-10-01"] * 3),
                         "is_lost": [0, 0, 1, 1, 1, 0],
                         "day_of_week": [0, 0, 1, 2, 2, 3]})
    for col in CATEGORICAL_FEATURES:
        data[col] = ["x", "x", "y", "x", "x", "z"]
    for col in NUMERIC_FEATURES:
        data[col] = [1., 1., 2., 3., 3., 4.]
    return data


class GPTests(unittest.TestCase):
    def test_temporal_encoding_ignores_current_and_future_labels(self):
        data = frame()
        # Compare unscaled temporal encodings: scaling may see all train features,
        # but no row's target-encoding value may see its own month's labels.
        encoder = Encoder()
        encoded = encoder.fit_transform(data)
        unscaled = (encoded.T * encoder.spread + encoder.center)
        changed = data.copy()
        changed.loc[3:, "is_lost"] = 1 - changed.loc[3:, "is_lost"]
        other = Encoder()
        reencoded = other.fit_transform(changed)
        other_unscaled = reencoded.T * other.spread + other.center
        np.testing.assert_allclose(unscaled[:, 6:12], other_unscaled[:, 6:12])
        np.testing.assert_allclose(unscaled[:3, 6:12], .5)

    def test_holdout_labels_never_enter_features(self):
        train = frame()
        encoder = Encoder()
        encoder.fit_transform(train)
        holdout = train.copy()
        before = encoder.transform(holdout)
        holdout["is_lost"] = 1 - holdout["is_lost"]
        np.testing.assert_array_equal(before, encoder.transform(holdout))

    def test_group_may_not_cross_encoding_month(self):
        data = frame()
        data.loc[3, "SHIPMENT_ID"] = "a"
        with self.assertRaises(ValueError):
            Encoder().fit_transform(data)

    def test_ablation_excludes_domain_and_unknowns_finite(self):
        data = frame()
        encoder = Encoder(include_domain=False)
        encoder.fit_transform(data)
        self.assertFalse(any("DOM_DOMAIN_ID" in n for n in encoder.names))
        holdout = data.copy()
        for col in CATEGORICAL_FEATURES:
            holdout[col] = "NEVER_SEEN"
        holdout["RATIO_SVC_MES_ANTERIOR"] = np.nan
        self.assertTrue(np.isfinite(encoder.transform(holdout)).all())

    def test_f1_threshold_matches_exhaustive_search(self):
        y = np.array([0, 1, 1, 0, 1, 0, 1])
        scores = np.array([.1, .2, .2, .3, .6, .7, .9])
        chosen = exact_f1_threshold(y, scores)
        self.assertAlmostEqual(f1_score(y, scores >= chosen),
                               max(f1_score(y, scores >= t) for t in np.unique(scores)))

    def test_constant_tree_broadcasts(self):
        np.testing.assert_array_equal(predict_tree("0.5", np.ones((2, 8))), np.full(8, .5))

    def test_encoder_roundtrip(self):
        data = frame()
        encoder = Encoder()
        encoder.fit_transform(data)
        restored = pickle.loads(pickle.dumps(encoder))
        np.testing.assert_array_equal(encoder.transform(data), restored.transform(data))


if __name__ == "__main__":
    unittest.main()
