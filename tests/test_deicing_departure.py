import unittest
from datetime import datetime, timezone, timedelta

from rampflow.deicing import DeicingManager, Pad, HoldoverBand
from rampflow.runway_slots import RunwaySlotManager, Runway, SlotRequest
from rampflow.slot_coordination import SlotCoordinator, SlotRequest as CoordRequest
from rampflow.deicing_departure import DeicingDeparturePlanner
from rampflow.models import Decision
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def make_deicing(bays=2):
    pads = [Pad("P1", bays, frozenset({"I"}), 80.0, 50.0)]
    bands = [HoldoverBand("snow", "I", -10, 0, 40, 60)]
    return DeicingManager(pads, bands)


def make_runway(wake=None):
    table = wake or {("H", "L"): 180, ("L", "H"): 45, ("*", "M"): 60}
    return RunwaySlotManager([Runway("09L", frozenset({"D"}), 5, 20, 60)], table)


class DeicingBayTests(unittest.TestCase):
    def test_second_bay_runs_concurrently(self):
        m = make_deicing(bays=2)
        t1 = m.schedule("F1", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        t2 = m.schedule("F2", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        t3 = m.schedule("F3", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        self.assertEqual(t1.start, D(8))
        self.assertEqual(t2.start, D(8))  # second bay, not queued behind F1
        self.assertEqual(t3.start, D(8, 6))  # both bays busy, queues

    def test_cancel_frees_exactly_one_bay(self):
        m = make_deicing(bays=2)
        m.schedule("F1", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        m.schedule("F2", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        m.schedule("F3", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        self.assertTrue(m.cancel("F1"))
        t4 = m.schedule("F4", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        self.assertEqual(t4.start, D(8))  # F1's bay reusable, F2's window untouched
        t5 = m.schedule("F5", "P1", D(8), 36.0, "I", 100, "snow", -5.0, 100.0)
        self.assertEqual(t5.start, D(8, 6))

    def test_pad_must_have_a_bay(self):
        with self.assertRaises(ValueError):
            make_deicing(bays=0)


class WakeSeparationTests(unittest.TestCase):
    def test_order_is_significant(self):
        rwy = make_runway()
        self.assertEqual(rwy._sep("H", "L"), 180)
        self.assertEqual(rwy._sep("L", "H"), 45)

    def test_missing_pair_falls_back_to_safe_default_not_reversed(self):
        rwy = make_runway(wake={("L", "H"): 45, ("*", "L"): 120})
        self.assertEqual(rwy._sep("H", "L"), 120)  # never the reversed 45s

    def test_committed_sequence_enforces_direction(self):
        rwy = make_runway()
        rwy.commit(rwy.find(SlotRequest("FH", "departure", "H", D(10), D(10), D(11))))
        light = rwy.find(SlotRequest("FL", "departure", "L", D(10), D(10), D(11)))
        self.assertGreaterEqual((light.slot - D(10)).total_seconds(), 180)

        rwy2 = make_runway()
        rwy2.commit(rwy2.find(SlotRequest("FL", "departure", "L", D(10), D(10), D(11))))
        heavy = rwy2.find(SlotRequest("FH", "departure", "H", D(10), D(10), D(11)))
        gap = (heavy.slot - D(10)).total_seconds()
        self.assertGreaterEqual(gap, 45)
        self.assertLess(gap, 180)  # not over-separated by the reversed pair

    def test_release_frees_the_slot(self):
        rwy = make_runway()
        rwy.commit(rwy.find(SlotRequest("FH", "departure", "H", D(10), D(10), D(11))))
        self.assertTrue(rwy.release("FH"))
        self.assertFalse(rwy.release("FH"))
        again = rwy.find(SlotRequest("FL", "departure", "L", D(10), D(10), D(11)))
        self.assertEqual(again.slot, D(10))  # no phantom separation from the released use


class SlotCoordinatorWakeTests(unittest.TestCase):
    def test_committed_wake_class_drives_separation(self):
        c = SlotCoordinator(["09L"], base_separation_seconds=60)
        c.commit(c.allocate(CoordRequest("H1", D(10), 0, 0, required_runway="09L", wake_class="H")))
        light = c.allocate(CoordRequest("L1", D(10), 0, 10, required_runway="09L", wake_class="L"))
        c.commit(light)
        self.assertGreaterEqual((light.time - D(10)).total_seconds(), 150)  # 60 base + 90 H->L


class JointPlanTests(unittest.TestCase):
    def make_planner(self, **kw):
        deicing = make_deicing()
        runway = make_runway()
        return deicing, runway, DeicingDeparturePlanner(deicing, runway, taxi_out_min=10, **kw)

    def plan_args(self, fid):
        return dict(flight_id=fid, pad_id="P1", ready=D(8), wingspan_m=36.0, fluid_type="I",
                    concentration=100, precip="snow", temp_c=-5.0, fluid_liters=100.0, wake_category="M")

    def test_pairs_earliest_bay_window_with_in_holdover_slot(self):
        deicing, _, planner = self.make_planner()
        plan = planner.plan(**self.plan_args("F1"))
        self.assertEqual(plan.treatment.start, D(8))
        self.assertEqual(plan.departure, D(8, 16))  # treatment end + taxi-out
        self.assertEqual(plan.expires_at, D(8, 46))
        self.assertEqual(plan.margin_min, 30)
        self.assertTrue(deicing.departure_status("F1", plan.departure).accepted)

    def test_margin_floor_is_a_real_tradeoff(self):
        _, _, planner = self.make_planner()
        plan = planner.plan(**self.plan_args("F1"), min_margin_min=30)
        self.assertGreaterEqual(plan.margin_min, 30)
        with self.assertRaises(ValueError):
            planner.plan(**self.plan_args("F2"), min_margin_min=31)  # taxi+margin exceeds holdover

    def test_two_bays_double_joint_throughput(self):
        _, _, planner = self.make_planner()
        a = planner.plan(**self.plan_args("F1"))
        b = planner.plan(**self.plan_args("F2"))
        self.assertEqual(a.treatment.start, b.treatment.start)  # parallel bays
        self.assertNotEqual(a.departure, b.departure)  # runway still serialised safely
        self.assertGreaterEqual((b.departure - a.departure).total_seconds(), 60)

    def test_impossible_runway_window_holds_nothing(self):
        deicing, runway, planner = self.make_planner(max_attempts=3)
        with self.assertRaises(RuntimeError):
            planner.plan(**self.plan_args("F1"), ctot=D(23))  # CTOT unreachable from any window
        self.assertEqual(deicing.treatments, {})
        self.assertEqual(runway._uses["09L"], [])
        self.assertEqual(planner.plans, {})

    def test_rejected_runway_commit_rolls_back_deicing(self):
        deicing, runway, planner = self.make_planner()
        runway.commit = lambda use: Decision(False, "slot-stale", "runway sequence changed")
        with self.assertRaises(RuntimeError):
            planner.plan(**self.plan_args("F1"))
        self.assertEqual(deicing.treatments, {})  # bay not left occupied
        self.assertEqual(planner.plans, {})

    def test_expired_plan_releases_bay_and_slot(self):
        deicing, runway, planner = self.make_planner()
        planner.plan(**self.plan_args("F1"))
        self.assertEqual(planner.release_expired(D(8, 46)), ())  # still valid at expiry
        self.assertEqual(planner.release_expired(D(8, 47)), ("F1",))
        self.assertEqual(planner.plans, {})
        self.assertEqual(deicing.treatments, {})
        self.assertEqual(runway._uses["09L"], [])
        # resources genuinely reusable: an identical request succeeds afterwards
        deicing2, runway2, planner2 = self.make_planner()
        again = planner2.plan(**self.plan_args("F9"))
        self.assertEqual(again.departure, D(8, 16))

    def test_cancel_releases_both_resources(self):
        deicing, runway, planner = self.make_planner()
        planner.plan(**self.plan_args("F1"))
        self.assertTrue(planner.cancel("F1"))
        self.assertFalse(planner.cancel("F1"))
        self.assertEqual(deicing.treatments, {})
        self.assertEqual(runway._uses["09L"], [])


class DeicingRunwayWorkflowTests(unittest.TestCase):
    def test_event_published_only_after_domain_success(self):
        control = RampWorkflowControl()

        def boom():
            raise RuntimeError("domain rejected")

        with self.assertRaises(RuntimeError):
            control.commit_deicing_runway_slot("op-1", "F1", D(10), boom)
        self.assertEqual(control.stream.head, 0)
        self.assertEqual(control.receipts, {})

    def test_success_is_idempotent(self):
        control = RampWorkflowControl()
        calls = []
        first = control.commit_deicing_runway_slot("op-2", "F1", D(10), lambda: calls.append(1) or "ok")
        second = control.commit_deicing_runway_slot("op-2", "F1", D(10), lambda: calls.append(2) or "dup")
        self.assertIs(first, second)
        self.assertEqual(calls, [1])
        self.assertEqual(control.stream.head, 1)

    def test_rejected_requirement_blocks_commit(self):
        control = RampWorkflowControl()
        denied = Decision(False, "holdover-expired", "plan expired before commit")
        with self.assertRaises(WorkflowRejected):
            control.commit_deicing_runway_slot("op-3", "F1", D(10), lambda: "ok", requirements=(denied,))
        self.assertEqual(control.stream.head, 0)


if __name__ == "__main__":
    unittest.main()
