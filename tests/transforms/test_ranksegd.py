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

from monai.transforms import RankSegd
from monai.utils import optional_import

_, has_rankseg = optional_import("rankseg")


@unittest.skipUnless(has_rankseg, "Requires rankseg")
class TestRankSegd(unittest.TestCase):
    def test_single_key(self):
        transform = RankSegd(keys="pred", metric="iou", mode="multiclass", output_mode="multiclass")
        data = {"pred": torch.softmax(torch.rand((1, 3, 10, 9), dtype=torch.float), dim=1)}

        out = transform(data)
        self.assertIn("pred", out)
        self.assertEqual(tuple(out["pred"].shape[-2:]), (10, 9))

    def test_with_gt_key(self):
        transform = RankSegd(
            keys="pred",
            gt_keys="label",
            metric="dice",
            mode="multiclass",
            output_mode="multiclass",
        )
        data = {
            "pred": torch.softmax(torch.rand((1, 2, 7, 6), dtype=torch.float), dim=1),
            "label": torch.randint(0, 2, size=(1, 7, 6)),
        }

        out = transform(data)
        self.assertEqual(tuple(out["pred"].shape[-2:]), (7, 6))


if __name__ == "__main__":
    unittest.main()
