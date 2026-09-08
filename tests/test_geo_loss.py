import math
import unittest

import numpy as np
import torch

from losses.geo_loss import (
    GeoLoss,
    bezier_polyline,
    build_reference,
    contour_prf,
    grad_norm_ratio,
    point_to_segments_dist,
)


CANVAS_SIZE = 224
CENTER = 112.0
RADIUS = 70.0


def cubic_arc(mid_degrees, span_degrees, radius):
    start = math.radians(mid_degrees - span_degrees / 2.0)
    end = math.radians(mid_degrees + span_degrees / 2.0)
    delta = end - start
    handle = 4.0 / 3.0 * math.tan(delta / 4.0)

    p0 = np.array([
        CENTER + radius * math.cos(start),
        CENTER + radius * math.sin(start),
    ])
    p3 = np.array([
        CENTER + radius * math.cos(end),
        CENTER + radius * math.sin(end),
    ])
    p1 = p0 + handle * radius * np.array([
        -math.sin(start),
        math.cos(start),
    ])
    p2 = p3 - handle * radius * np.array([
        -math.sin(end),
        math.cos(end),
    ])
    return np.stack((p0, p1, p2, p3))


def make_reference():
    yy, xx = np.ogrid[:CANVAS_SIZE, :CANVAS_SIZE]
    radius = np.sqrt((xx - CENTER) ** 2 + (yy - CENTER) ** 2)
    circle = np.abs(radius - RADIUS) <= 0.5
    diameter = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    diameter[int(CENTER), 42:183] = True
    return build_reference(
        [(diameter, 1.0), (circle, 3.0)],
        (CANVAS_SIZE, CANVAS_SIZE),
        min_len=15,
        device="cpu",
    )


def make_offset_strokes(span=18.0):
    strokes = [
        cubic_arc(angle, span, RADIUS + 25.0)
        for angle in np.arange(0.0, 360.0, 30.0)
    ]
    for center_x in (75.0, 100.0, 124.0, 149.0):
        strokes.append(np.array([
            [center_x - 5.0, CENTER + 25.0],
            [center_x - 5.0 / 3.0, CENTER + 25.0],
            [center_x + 5.0 / 3.0, CENTER + 25.0],
            [center_x + 5.0, CENTER + 25.0],
        ]))
    return torch.tensor(np.stack(strokes), dtype=torch.float32)


def make_ceiling_strokes():
    strokes = [
        cubic_arc(
            (index + 0.5) * 360.0 / 14.0,
            360.0 / 14.0,
            RADIUS,
        )
        for index in range(14)
    ]
    strokes.extend([
        np.array([
            [42.0, CENTER],
            [65.333, CENTER],
            [88.667, CENTER],
            [112.0, CENTER],
        ]),
        np.array([
            [112.0, CENTER],
            [135.333, CENTER],
            [158.667, CENTER],
            [182.0, CENTER],
        ]),
    ])
    return torch.tensor(np.stack(strokes), dtype=torch.float32)


def stroke_length(points):
    vertices = bezier_polyline(points, M=64)
    return float(torch.linalg.vector_norm(
        vertices[:, 1:] - vertices[:, :-1],
        dim=-1,
    ).sum())


class TestGeoLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = make_reference()
        cls._results = None
        cls._initial_f = None

    @classmethod
    def run_ablation(cls):
        if cls._results is not None:
            return cls._results

        initial = make_offset_strokes()
        cls._initial_f = contour_prf(initial, cls.ref, sigma=4.0)["f"]
        results = {}
        for name, w_pos, w_cov in (
            ("cov", 0.0, 1.0),
            ("pos", 1.0, 0.0),
            ("both", 1.0, 1.0),
        ):
            torch.manual_seed(0)
            points = initial.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([points], lr=1.0)
            criterion = GeoLoss(
                cls.ref,
                T=400,
                M=16,
                w_pos=w_pos,
                w_cov=w_cov,
                w_tan=0.0,
                w_anchor=0.0,
                tau0=16.0,
                tau1=2.0,
                cov_subsample=4096,
            )
            for step in range(400):
                optimizer.zero_grad()
                loss, _ = criterion(points, step=step)
                loss.backward()
                optimizer.step()
            results[name] = contour_prf(
                points.detach(),
                cls.ref,
                sigma=4.0,
            )
            results[name]["length"] = stroke_length(points.detach())
        cls._results = results
        return results

    def test_t1_shrink(self):
        generator = torch.Generator().manual_seed(123)
        points = None
        for _ in range(5):
            centers = CENTER + (
                torch.rand((16, 1, 2), generator=generator) - 0.5
            ) * 190.0
            directions = torch.randn((16, 1, 2), generator=generator)
            directions /= torch.linalg.vector_norm(
                directions,
                dim=-1,
                keepdim=True,
            )
            offsets = torch.linspace(-15.0, 15.0, 4).view(1, 4, 1)
            points = centers + offsets * directions

        criterion = GeoLoss(
            self.ref,
            T=400,
            w_tan=1.0,
            w_anchor=0.0,
            cov_subsample=0,
        )
        logs = []
        for scale in (1.0, 0.5, 0.2, 0.05):
            scaled = CENTER + (points - CENTER) * scale
            _, values = criterion(scaled, step=0)
            logs.append(values)

        coverage = [values["cov"] for values in logs]
        self.assertTrue(all(
            left < right
            for left, right in zip(coverage, coverage[1:])
        ))
        self.assertLess(logs[1]["pos"], logs[0]["pos"])
        self.assertLess(logs[1]["tan"], logs[0]["tan"])

    def test_t2_recovery(self):
        results = self.run_ablation()
        self.assertEqual(round(self._initial_f, 3), 0.0)
        self.assertGreaterEqual(results["both"]["f"], 0.98)

    def test_t3_ablation(self):
        results = self.run_ablation()
        cov = results["cov"]
        pos = results["pos"]
        both = results["both"]

        self.assertGreaterEqual(cov["recall"], 0.99)
        self.assertLessEqual(cov["precision"], 0.94)
        self.assertGreater(cov["length"], both["length"])

        self.assertGreaterEqual(pos["precision"], 0.98)
        self.assertLessEqual(pos["recall"], 0.94)
        self.assertLess(pos["length"], both["length"])

        self.assertGreaterEqual(both["f"], 0.98)
        self.assertGreater(both["f"], cov["f"])
        self.assertGreater(both["f"], pos["f"])

    def test_t4_metric_floor_and_ceiling(self):
        blank = torch.empty((0, 4, 2), dtype=torch.float32)
        self.assertEqual(contour_prf(blank, self.ref)["f"], 0.0)

        generator = torch.Generator().manual_seed(1)
        centers = 42.0 + torch.rand(
            (16, 1, 2),
            generator=generator,
        ) * 140.0
        directions = torch.randn((16, 1, 2), generator=generator)
        directions /= torch.linalg.vector_norm(
            directions,
            dim=-1,
            keepdim=True,
        )
        random_strokes = centers + torch.linspace(
            -20.0,
            20.0,
            4,
        ).view(1, 4, 1) * directions
        random_f = contour_prf(random_strokes, self.ref)["f"]
        self.assertGreaterEqual(random_f, 0.2)
        self.assertLessEqual(random_f, 0.4)

        ceiling_f = contour_prf(
            make_ceiling_strokes(),
            self.ref,
        )["f"]
        self.assertGreaterEqual(ceiling_f, 0.9)

    def test_t5_polyline_accuracy(self):
        curve = torch.tensor(
            cubic_arc(45.0, 90.0, RADIUS),
            dtype=torch.float64,
        ).unsqueeze(0)
        coarse = bezier_polyline(curve, M=16)
        dense = bezier_polyline(curve, M=4096)[0]
        error = point_to_segments_dist(dense, coarse, chunk=512)
        self.assertLess(float(error.max()), 0.2)


    def test_step_zero_is_independent_of_geo_hold(self):
        points = make_offset_strokes()
        logs = []
        metrics = []
        for geo_hold in (0.5, 0.2):
            criterion = GeoLoss(
                self.ref, T=400, w_anchor=0.0, geo_hold=geo_hold,
                cov_subsample=0,
            )
            _, values = criterion(points, step=0)
            logs.append(values)
            metrics.append(contour_prf(points, self.ref, sigma=4.0))
        for name in ("pos", "cov"):
            self.assertEqual(logs[0][name], logs[1][name])
        for name in ("precision", "recall"):
            self.assertEqual(metrics[0][name], metrics[1][name])

    def test_gradient_diagnostics(self):
        points = torch.tensor([1.0, -2.0], requires_grad=True)
        loss_sds = points.square().sum()
        loss_geo = -points.square().sum()
        rng_before = torch.random.get_rng_state().clone()

        logs = grad_norm_ratio(points, loss_sds, loss_geo)

        self.assertAlmostEqual(logs["g_sds"], logs["g_geo"])
        self.assertAlmostEqual(logs["ratio"], 1.0)
        self.assertAlmostEqual(logs["cos"], -1.0)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertIsNone(points.grad)
        (loss_sds + loss_geo).backward()
        self.assertTrue(torch.equal(points.grad, torch.zeros_like(points)))

    def test_gradient_diagnostics_with_disabled_geometry(self):
        points = torch.tensor([1.0, -2.0], requires_grad=True)
        loss_sds = points.square().sum()
        loss_geo = loss_sds.detach().new_zeros(())

        logs = grad_norm_ratio(points, loss_sds, loss_geo)

        self.assertEqual(logs["g_geo"], 0.0)
        self.assertEqual(logs["ratio"], 0.0)
        self.assertEqual(logs["cos"], 0.0)


if __name__ == "__main__":
    unittest.main()
