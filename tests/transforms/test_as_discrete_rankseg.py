# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import torch

from monai.transforms import AsDiscrete, AsDiscreted
from monai.utils.module import optional_import

RankSEG, has_rankseg = optional_import("rankseg")


class TestAsDiscreteRankSeg(unittest.TestCase):
    """Tests for RankSEG integration in AsDiscrete."""

    def test_rankseg_missing_dependency(self):
        """Test that ModuleNotFoundError is raised when rankseg is not installed."""
        if has_rankseg:
            self.skipTest("rankseg is installed, skipping missing dependency test")
        
        probs = torch.rand(3, 4, 5)
        transform = AsDiscrete(rankseg=True)
        
        with self.assertRaises(ModuleNotFoundError):
            transform(probs)

    def test_rankseg_incompatible_with_argmax(self):
        """Test that ValueError is raised when both argmax and rankseg are True."""
        probs = torch.rand(3, 4, 5)
        # Test with both set at init
        transform = AsDiscrete(argmax=True, rankseg=True)
        with self.assertRaises(ValueError):
            transform(probs)
        
        # Test with one set at init and one at call
        transform2 = AsDiscrete(argmax=True)
        with self.assertRaises(ValueError):
            transform2(probs, rankseg=True)

    def test_rankseg_incompatible_with_threshold(self):
        """Test that ValueError is raised when both threshold and rankseg are set."""
        probs = torch.rand(3, 4, 5)
        # Test with both set at init
        transform = AsDiscrete(threshold=0.5, rankseg=True)
        with self.assertRaises(ValueError):
            transform(probs)
        
        # Test with one set at init and one at call
        transform2 = AsDiscrete(rankseg=True)
        with self.assertRaises(ValueError):
            transform2(probs, threshold=0.5)

    def test_rankseg_incompatible_with_rounding(self):
        """Test that ValueError is raised when both rounding and rankseg are set."""
        probs = torch.rand(3, 4, 5)
        # Test with both set at init
        transform = AsDiscrete(rounding="torchrounding", rankseg=True)
        with self.assertRaises(ValueError):
            transform(probs)
        
        # Test with one set at init and one at call
        transform2 = AsDiscrete(rankseg=True)
        with self.assertRaises(ValueError):
            transform2(probs, rounding="torchrounding")

    def test_existing_argmax_behavior_unchanged(self):
        """Test that existing argmax behavior is unchanged."""
        probs = torch.rand(3, 4, 5)
        transform = AsDiscrete(argmax=True)
        result = transform(probs)
        
        # Should have shape (1, 4, 5) with keepdim=True (default)
        self.assertEqual(result.shape, (1, 4, 5))
    
    def test_existing_threshold_behavior_unchanged(self):
        """Test that existing threshold behavior is unchanged."""
        probs = torch.tensor([[[0.0, 0.5], [0.8, 3.0]]])
        transform = AsDiscrete(threshold=0.6)
        result = transform(probs)
        
        expected = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        self.assertTrue(torch.allclose(result.float(), expected))

    @unittest.skipUnless(has_rankseg, "rankseg is not installed")
    def test_rankseg_basic_2d(self):
        """Test RankSEG decoding with 2D multiclass probability input."""
        # Create a simple probability map
        probs = torch.rand(3, 8, 8)  # 3 classes, 8x8 spatial
        probs = probs / probs.sum(dim=0, keepdim=True)  # Normalize to probabilities
        
        transform = AsDiscrete(
            rankseg=True,
            rankseg_kwargs={"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}
        )
        result = transform(probs)
        
        # Output should be similar to argmax: (1, 8, 8) with keepdim=True
        self.assertEqual(result.shape[1:], (8, 8))
        # Output should contain valid class labels
        unique_labels = result.unique()
        self.assertTrue(all(label >= 0 for label in unique_labels))

    @unittest.skipUnless(has_rankseg, "rankseg is not installed")
    def test_rankseg_basic_3d(self):
        """Test RankSEG decoding with 3D multiclass probability input."""
        # Create a simple 3D probability map
        probs = torch.rand(3, 4, 4, 4)  # 3 classes, 4x4x4 spatial
        probs = probs / probs.sum(dim=0, keepdim=True)  # Normalize to probabilities
        
        transform = AsDiscrete(
            rankseg=True,
            rankseg_kwargs={"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}
        )
        result = transform(probs)
        
        # Output should be similar to argmax: (1, 4, 4, 4) with keepdim=True
        self.assertEqual(result.shape[1:], (4, 4, 4))
        # Output should contain valid class labels
        unique_labels = result.unique()
        self.assertTrue(all(label >= 0 for label in unique_labels))

    @unittest.skipUnless(has_rankseg, "rankseg is not installed")
    def test_rankseg_keepdim_false(self):
        """Test RankSEG decoding with keepdim=False."""
        probs = torch.rand(3, 8, 8)
        probs = probs / probs.sum(dim=0, keepdim=True)
        
        transform = AsDiscrete(
            rankseg=True,
            rankseg_kwargs={"metric": "dice", "solver": "RMA", "output_mode": "multiclass"},
            keepdim=False
        )
        result = transform(probs)
        
        # With keepdim=False, output should be (8, 8)
        self.assertEqual(result.shape, (8, 8))


class TestAsDiscretedRankSeg(unittest.TestCase):
    """Tests for RankSEG integration in AsDiscreted."""

    def test_rankseg_missing_dependency_dict(self):
        """Test that ModuleNotFoundError is raised when rankseg is not installed."""
        if has_rankseg:
            self.skipTest("rankseg is installed, skipping missing dependency test")
        
        data = {"pred": torch.rand(3, 4, 5)}
        transform = AsDiscreted(keys=["pred"], rankseg=True)
        
        with self.assertRaises(ModuleNotFoundError):
            transform(data)

    def test_existing_dict_behavior_unchanged(self):
        """Test that existing AsDiscreted behavior is unchanged."""
        data = {"pred": torch.rand(3, 4, 5)}
        transform = AsDiscreted(keys=["pred"], argmax=True)
        result = transform(data)
        
        self.assertEqual(result["pred"].shape, (1, 4, 5))

    def test_dict_allow_missing_keys(self):
        """Test that allow_missing_keys is respected with rankseg option."""
        data = {"other": torch.rand(3, 4, 5)}
        transform = AsDiscreted(keys=["pred"], rankseg=False, allow_missing_keys=True)
        result = transform(data)
        
        # Should not raise error and return original data
        self.assertIn("other", result)

    @unittest.skipUnless(has_rankseg, "rankseg is not installed")
    def test_rankseg_dict_basic(self):
        """Test RankSEG decoding in dictionary format."""
        data = {"pred": torch.rand(3, 8, 8)}
        data["pred"] = data["pred"] / data["pred"].sum(dim=0, keepdim=True)
        
        transform = AsDiscreted(
            keys=["pred"],
            rankseg=True,
            rankseg_kwargs={"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}
        )
        result = transform(data)
        
        self.assertEqual(result["pred"].shape[1:], (8, 8))


if __name__ == "__main__":
    unittest.main()
