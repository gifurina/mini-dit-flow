"""配置边界：非法参数和不兼容 checkpoint 必须在启动前被拒绝。"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        self.config["device"] = "cpu"

    def load(self, config, mode="train"):
        filename = self.directory / "config.yaml"
        filename.write_text(yaml.safe_dump(config))
        return load_config(filename, mode)

    def checkpoint(self, epoch=1):
        filename = self.directory / "checkpoint.pt"
        filename.touch()
        return str(filename), {"config": copy.deepcopy(self.config), "epoch": epoch}

    def test_valid_training_needs_no_existing_sample_checkpoint(self):
        config, checkpoints = self.load(self.config)
        self.assertEqual(config, self.config)
        self.assertEqual(checkpoints, {})

    def test_invalid_parameters(self):
        cases = [
            ("seed", None, True), ("num_threads", None, 0),
            ("data", "batch_size", 0), ("data", "val_size", 60000),
            ("data", "num_workers", -1), ("data", "download", "false"),
            ("model", "patch_size", 3), ("model", "emb_size", 126),
            ("model", "num_heads", 3), ("model", "img_size", 32),
            ("train", "learning_rate", float("nan")), ("train", "sample_every", 0),
            ("train", "epochs", 0), ("train", "amp", 1),
            ("sample", "steps", 0), ("sample", "class_id", 10),
            ("sample", "num_samples", 0), ("sample", "compare_steps", []),
            ("sample", "compare_steps", [10, -1]),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                config = copy.deepcopy(self.config)
                if key is None:
                    config[section] = value
                else:
                    config[section][key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_missing_and_unknown_fields(self):
        for change in ("missing", "unknown"):
            config = copy.deepcopy(self.config)
            if change == "missing":
                del config["train"]["epochs"]
            else:
                config["train"]["epoch"] = 30
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(config)

    def test_unavailable_cuda(self):
        self.config["device"] = "cuda"
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA 不可用"):
                self.load(self.config)

    def test_sampling_checkpoint_and_model_compatibility(self):
        self.config["sample"]["checkpoint"] = str(self.directory / "missing.pt")
        with self.assertRaisesRegex(ValueError, "文件不存在"):
            self.load(self.config, "sample")
        filename, checkpoint = self.checkpoint()
        self.config["sample"]["checkpoint"] = filename
        with patch("config.torch.load", return_value=checkpoint) as read:
            _, checkpoints = self.load(self.config, "sample")
            self.assertIs(checkpoints["checkpoint"], checkpoint)
            read.assert_called_once()
            self.config["model"]["depth"] += 1
            with self.assertRaisesRegex(ValueError, "模型配置不一致"):
                self.load(self.config, "sample")

    def test_resume_requires_more_epochs(self):
        filename, checkpoint = self.checkpoint(epoch=30)
        self.config["train"]["resume"] = filename
        with patch("config.torch.load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "已完成轮数"):
                self.load(self.config)


if __name__ == "__main__":
    unittest.main()
