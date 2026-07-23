import inspect
import unittest

from multiplex.hub import Hub
from multiplex.scheduler import DynamicDepthController, Req, Scheduler
from multiplex.server import parse_args as parse_server_args, serve
from try_vision import parse_args as parse_vision_args


def feed(controller, accepted):
    decision = None
    for value in accepted:
        decision = controller.observe(value)
    return decision


class DynamicDepthTests(unittest.TestCase):
    def test_all_runtime_entry_points_default_to_dynamic_depth_three(self):
        scheduler_defaults = inspect.signature(Scheduler).parameters
        hub_defaults = inspect.signature(Hub).parameters
        serve_defaults = inspect.signature(serve).parameters

        self.assertEqual(scheduler_defaults["k"].default, 3)
        self.assertIs(scheduler_defaults["dynamic_depth"].default, True)
        self.assertEqual(hub_defaults["k"].default, 3)
        self.assertIs(hub_defaults["dynamic_depth"].default, True)
        self.assertEqual(serve_defaults["depth"].default, 3)
        self.assertIs(serve_defaults["dynamic_depth"].default, True)

        server_args = parse_server_args([])
        self.assertEqual(server_args.depth, 3)
        self.assertIs(server_args.dynamic_depth, True)

        vision_args = parse_vision_args([])
        self.assertEqual(vision_args.depth, 3)
        self.assertIs(vision_args.dynamic_depth, True)

    def test_steps_down_one_level_at_a_time(self):
        controller = DynamicDepthController(
            3,
            window=4,
            min_samples=4,
            down_threshold=0.5,
            up_threshold=0.8,
            retry_cooldown=8,
        )

        decision = feed(controller, [0, 1, 0, 1])
        self.assertEqual(decision.reason, "low_acceptance")
        self.assertEqual((decision.previous, decision.current), (3, 2))

        decision = feed(controller, [0, 0, 1, 1])
        self.assertEqual(decision.reason, "low_acceptance")
        self.assertEqual((decision.previous, decision.current), (2, 1))

        decision = feed(controller, [0, 0, 0, 0])
        self.assertEqual(decision.current, 1)
        self.assertIsNone(decision.reason)

    def test_retries_higher_depth_after_cooldown(self):
        controller = DynamicDepthController(
            3,
            window=4,
            min_samples=2,
            down_threshold=0.5,
            up_threshold=0.8,
            retry_cooldown=4,
        )
        feed(controller, [0, 0])
        self.assertEqual(controller.current, 2)

        # Full D2 acceptance alone cannot immediately undo the downshift.
        decision = feed(controller, [2, 2])
        self.assertEqual(decision.current, 2)
        self.assertEqual(controller.cooldown, 2)

        decision = feed(controller, [2, 2])
        self.assertEqual(decision.reason, "high_acceptance")
        self.assertEqual((decision.previous, decision.current), (2, 3))

    def test_hysteresis_holds_middle_rate(self):
        controller = DynamicDepthController(
            3,
            window=8,
            min_samples=8,
            down_threshold=0.5,
            up_threshold=0.8,
        )

        decision = feed(controller, [3, 0, 3, 0, 3, 0, 3, 3])
        self.assertAlmostEqual(decision.full_acceptance, 0.625)
        self.assertEqual(decision.current, 3)
        self.assertIsNone(decision.reason)

    def test_reset_can_restart_at_max(self):
        controller = DynamicDepthController(
            3,
            window=2,
            min_samples=2,
            down_threshold=0.5,
            up_threshold=0.8,
        )
        feed(controller, [0, 0])
        self.assertEqual(controller.current, 2)

        controller.reset(restart_at_max=False)
        self.assertEqual(controller.current, 2)
        self.assertEqual(controller.samples, 0)

        controller.reset(restart_at_max=True)
        self.assertEqual(controller.current, 3)
        self.assertEqual(controller.samples, 0)
        self.assertEqual(controller.cooldown, 0)

    def test_acceptance_rates_use_per_depth_trial_counts(self):
        req = Req(0, [1], 10)
        req.accept_counts = [8, 4, 1]
        req.accept_trials_by_depth = [10, 5, 2]

        self.assertEqual(Scheduler._acceptance_rates(req), [0.8, 0.8, 0.5])

    def test_rejects_invalid_threshold_order(self):
        with self.assertRaisesRegex(ValueError, "down_threshold"):
            DynamicDepthController(3, down_threshold=0.9, up_threshold=0.8)


if __name__ == "__main__":
    unittest.main()
