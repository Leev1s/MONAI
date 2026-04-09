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
from parameterized import parameterized

from monai.transforms import RankSeg
from monai.utils import optional_import

_, has_rankseg = optional_import("rankseg")


@unittest.skipUnless(has_rankseg, "Requires rankseg")
class TestRankSeg(unittest.TestCase):
    @parameterized.expand([
        [{"mode": "multiclass", "output_mode": "multiclass"}],
        [{"mode": "multilabel", "output_mode": "multilabel"}],
    ])
    def test_shape(self, params):
        transform = RankSeg(metric="iou", solver="RMA", **params)
        probs = torch.rand((2, 3, 16, 12), dtype=torch.float)
        probs = torch.softmax(probs, dim=1)
        out = transform(probs)

        self.assertEqual(out.shape[0], probs.shape[0])
        self.assertEqual(tuple(out.shape[-2:]), tuple(probs.shape[-2:]))

    def test_with_gt(self):
        transform = RankSeg(metric="dice", mode="multiclass", output_mode="multiclass")
        probs = torch.rand((1, 4, 8, 8), dtype=torch.float)
        probs = torch.softmax(probs, dim=1)
        gt = torch.randint(0, 4, size=(1, 8, 8))
        out = transform(probs, gt=gt)

        self.assertEqual(out.shape[0], probs.shape[0])
        self.assertEqual(tuple(out.shape[-2:]), tuple(probs.shape[-2:]))


if __name__ == "__main__":
    unittest.main()
