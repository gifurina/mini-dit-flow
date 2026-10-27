"""使用解析答案检查流匹配方向、Euler 积分和完整模型的连接。"""

import unittest

import torch
from torch import nn

from flow import euler_sample, flow_matching_loss
from model import DiT


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_loss_uses_noise_to_image_velocity(self):
        images = torch.linspace(-1, 1, 2 * 28 * 28).reshape(2, 1, 28, 28)
        labels = torch.tensor([0, 9])
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(images.shape, generator=generator)
        times = torch.rand(2, generator=generator)
        t = times[:, None, None, None]
        expected_xt = (1 - t) * noise + t * images

        class Oracle(nn.Module):
            def forward(self, xt, time, classes):
                torch.testing.assert_close(xt, expected_xt)
                torch.testing.assert_close(time, times)
                torch.testing.assert_close(classes, labels)
                return images - noise

        loss = flow_matching_loss(Oracle(), images, labels, torch.Generator().manual_seed(7))
        self.assertEqual(loss.item(), 0.0)

    def test_euler_time_direction_and_mode(self):
        class TimeField(nn.Module):
            def forward(self, x, t, labels):
                self.assertions(x)
                return t[:, None, None, None].expand_as(x)

            def assertions(self, x):
                assert not self.training
                assert not torch.is_grad_enabled()

        model = TimeField()
        noise = torch.full((2, 1, 28, 28), 2.0, requires_grad=True)
        samples = euler_sample(model, noise, torch.tensor([0, 9]), steps=4)
        # 左端点 Euler：dt * (0 + 0.25 + 0.5 + 0.75) = 0.375。
        torch.testing.assert_close(samples, noise + 0.375)
        self.assertTrue(model.training)
        self.assertFalse(samples.requires_grad)
        self.assertTrue(torch.all(noise == 2))

    def test_complete_dit_shape_and_learning(self):
        torch.manual_seed(42)
        model = DiT()
        self.assertEqual(len({id(block) for block in model.blocks}), 4)
        images = torch.rand(2, 1, 28, 28) * 2 - 1
        labels = torch.tensor([0, 9])
        velocity = model(images, torch.tensor([0.25, 0.75]), labels)
        self.assertEqual(velocity.shape, images.shape)
        self.assertEqual(torch.count_nonzero(velocity).item(), 0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # 零初始化先学习输出投影和门控，随后梯度才能进入每层注意力。
        for _ in range(4):
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, images, labels, torch.Generator().manual_seed(9))
            loss.backward()
            optimizer.step()
        for block in model.blocks:
            self.assertTrue(torch.isfinite(block.attn.in_proj_weight.grad).all())
            self.assertGreater(block.attn.in_proj_weight.grad.abs().sum().item(), 0)
        samples = euler_sample(model, torch.randn_like(images), labels, steps=2)
        self.assertEqual(samples.shape, images.shape)
        self.assertTrue(torch.isfinite(samples).all())


if __name__ == "__main__":
    unittest.main()
