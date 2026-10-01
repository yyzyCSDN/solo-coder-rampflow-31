import unittest
from datetime import datetime, timedelta, timezone

from rampflow.deicing import DeicingManager, HoldoverBand, Pad
from rampflow.deicing_departure import (
    DeicingDeparturePlanner,
    JointPlan,
    JointPolicy,
    JointRequest,
    PlanState,
)
from rampflow.runway_slots import Runway, RunwaySlotManager, SlotRequest
from rampflow.timeutil import Window


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
BANDS = [HoldoverBand("snow", "T4", -20, 0, 15, 30)]
# directional wake table: pair key is (leader, follower)
WAKE = {
    ("H", "L"): 120,
    ("H", "M"): 90,
    ("M", "L"): 90,
    ("*", "H"): 60,
    ("*", "M"): 60,
    ("*", "L"): 60,
}


def pad(bays=2):
    return Pad("P1", bays, frozenset({"T4"}), 60.0, 100.0)


def runway_manager():
    return RunwaySlotManager(
        [Runway("09", frozenset({"09"}), 10, 30, 60)], WAKE, 30
    )


def deicing_manager(bays=2):
    return DeicingManager([pad(bays)], BANDS)


def joint_request(flight_id, target=T0, taxi=10):
    return JointRequest(
        flight_id, "L", T0, target, T0 + timedelta(hours=2), taxi,
        36.0, "T4", 100, "snow", -5.0, 100.0,
    )


class MultiBayDeicingTests(unittest.TestCase):
    def test_two_bays_serve_two_aircraft_simultaneously(self):
        dm = deicing_manager()
        a = dm.schedule("CA1", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        b = dm.schedule("CA2", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        self.assertEqual(a.start, T0)
        self.assertEqual(b.start, T0)
        self.assertNotEqual(a.bay, b.bay)

    def test_third_aircraft_queues_for_a_free_bay(self):
        dm = deicing_manager()
        a = dm.schedule("CA1", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        dm.schedule("CA2", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        c = dm.schedule("CA3", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        self.assertEqual(c.start, a.end)
        self.assertEqual(c.bay, a.bay)

    def test_single_bay_pad_still_serializes(self):
        dm = deicing_manager(bays=1)
        a = dm.schedule("CA1", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        b = dm.schedule("CA2", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        self.assertEqual(b.start, a.end)
        self.assertEqual(a.bay, b.bay)

    def test_cancel_frees_only_that_bay(self):
        dm = deicing_manager()
        a = dm.schedule("CA1", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        dm.schedule("CA2", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        self.assertTrue(dm.cancel("CA1"))
        c = dm.schedule("CA3", "P1", T0, 36, "T4", 100, "snow", -5, 100)
        self.assertEqual(c.bay, a.bay)
        self.assertEqual(c.start, T0)

    def test_pad_requires_positive_bays(self):
        with self.assertRaises(ValueError):
            Pad("P0", 0, frozenset({"T4"}), 60.0, 100.0)


class WakeSeparationDirectionTests(unittest.TestCase):
    def _gap(self, leader, follower):
        rm = runway_manager()
        first = rm.find(SlotRequest("A", "dep", leader, T0, T0, T0 + timedelta(hours=1)))
        rm.commit(first)
        second = rm.find(SlotRequest("B", "dep", follower, T0, T0, T0 + timedelta(hours=1)))
        rm.commit(second)
        return (second.slot - first.slot).total_seconds()

    def test_separation_is_directional(self):
        # key sorting used to swap leader and follower, under-separating H->L
        self.assertEqual(self._gap("H", "L"), 120)
        self.assertEqual(self._gap("M", "L"), 90)
        self.assertEqual(self._gap("L", "H"), 60)

    def test_resequence_uses_leader_to_follower_order(self):
        rm = runway_manager()
        h = rm.find(SlotRequest("H", "dep", "H", T0, T0, T0 + timedelta(hours=1)))
        rm.commit(h)
        l = rm.find(SlotRequest("L", "dep", "L", T0, T0, T0 + timedelta(hours=1)))
        rm.commit(l)
        updated = rm.resequence("09", ["H", "L"], T0)
        self.assertEqual((updated[1].slot - updated[0].slot).total_seconds(), 120)

    def test_release_returns_committed_slot(self):
        rm = runway_manager()
        use = rm.find(SlotRequest("L1", "dep", "L", T0, T0, T0 + timedelta(hours=1)))
        rm.commit(use)
        self.assertTrue(rm.release("L1"))
        self.assertIsNone(rm.conflict_for("09", "dep", "L", use.slot))
        self.assertFalse(rm.release("L1"))


class JointPlannerTests(unittest.TestCase):
    def test_plans_use_parallel_bays_and_enforce_runway_gap(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        target = T0 + timedelta(minutes=15)
        b1 = planner.book(planner.plan(joint_request("JX1", target)))
        b2 = planner.book(planner.plan(joint_request("JX2", target)))
        self.assertNotEqual(b1.bay, b2.bay)
        self.assertTrue(b1.treatment_window.overlaps(b2.treatment_window))
        self.assertGreaterEqual(
            (b2.runway_slot - b1.runway_slot).total_seconds(), 60
        )
        # a third aircraft queues behind a busy bay
        b3 = planner.book(planner.plan(joint_request("JX3", target)))
        self.assertGreaterEqual(
            b3.treatment_window.start,
            min(b1.treatment_window.end, b2.treatment_window.end),
        )

    def test_cancel_releases_bay_and_slot_for_reuse(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        target = T0 + timedelta(minutes=15)
        gone = planner.book(planner.plan(joint_request("GONE", target)))
        self.assertTrue(planner.cancel("GONE"))
        self.assertNotIn("GONE", dm.treatments)
        self.assertIsNone(rm.conflict_for("09", "dep", "L", gone.runway_slot))
        again = planner.book(planner.plan(joint_request("NEXT", target)))
        self.assertEqual(again.bay, gone.bay)
        self.assertEqual(again.treatment_window.start, gone.treatment_window.start)
        self.assertEqual(again.runway_slot, gone.runway_slot)

    def test_confirmed_holdover_expiry_releases_resources(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        plan = planner.book(planner.plan(joint_request("JX1")))
        planner.confirm_treatment("JX1", plan.treatment_window.end)
        freed = planner.purge_expired(plan.expires_at + timedelta(seconds=1))
        self.assertEqual(freed, ("JX1",))
        self.assertEqual(planner.plans["JX1"].state, PlanState.EXPIRED)
        self.assertNotIn("JX1", dm.treatments)
        self.assertIsNone(rm.conflict_for("09", "dep", "L", plan.runway_slot))

    def test_departure_after_expiry_is_rejected_and_releases(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        plan = planner.book(planner.plan(joint_request("JX1")))
        planner.confirm_treatment("JX1", plan.treatment_window.end)
        with self.assertRaises(RuntimeError):
            planner.depart("JX1", plan.expires_at + timedelta(minutes=1))
        self.assertEqual(planner.plans["JX1"].state, PlanState.EXPIRED)
        self.assertNotIn("JX1", dm.treatments)
        self.assertIsNone(rm.conflict_for("09", "dep", "L", plan.runway_slot))

    def test_unworked_proposed_plan_is_purged(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        plan = planner.book(planner.plan(joint_request("JX1")))
        freed = planner.purge_expired(plan.treatment_window.start + timedelta(seconds=1))
        self.assertEqual(freed, ("JX1",))
        self.assertNotIn("JX1", dm.treatments)
        self.assertIsNone(rm.conflict_for("09", "dep", "L", plan.runway_slot))

    def test_happy_path_departs_and_frees_bay(self):
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        plan = planner.book(planner.plan(joint_request("JX1")))
        planner.confirm_treatment("JX1", plan.treatment_window.end)
        departed = planner.depart("JX1", plan.runway_slot)
        self.assertEqual(departed.state, PlanState.DEPARTED)
        self.assertNotIn("JX1", dm.treatments)
        self.assertFalse(planner.cancel("JX1"))

    def test_extended_holdover_requires_inspection(self):
        dm, rm = deicing_manager(), runway_manager()
        for minute in range(21):  # saturate runway through 12:20
            t = T0 + timedelta(minutes=minute)
            use = rm.find(SlotRequest(f"L{minute}", "dep", "L", t, t, T0 + timedelta(hours=2)))
            rm.commit(use)
        planner = DeicingDeparturePlanner(
            dm, rm, JointPolicy(extended_holdover_penalty_min=0.0)
        )
        plan = planner.book(
            planner.plan(joint_request("JX1", T0 + timedelta(minutes=21), taxi=5))
        )
        self.assertEqual(plan.tier, "extended")
        planner.confirm_treatment("JX1", plan.treatment_window.end)
        with self.assertRaises(RuntimeError):
            planner.depart("JX1", plan.runway_slot)
        planner.record_inspection("JX1")
        self.assertEqual(
            planner.depart("JX1", plan.runway_slot).state, PlanState.DEPARTED
        )

    def test_policy_trades_throughput_against_safety_margin(self):
        def build(policy):
            dm, rm = deicing_manager(), runway_manager()
            for minute in range(21):
                t = T0 + timedelta(minutes=minute)
                use = rm.find(SlotRequest(f"L{minute}", "dep", "L", t, t, T0 + timedelta(hours=2)))
                rm.commit(use)
            planner = DeicingDeparturePlanner(dm, rm, policy)
            return planner.book(
                planner.plan(joint_request("JX1", T0 + timedelta(minutes=21), taxi=5))
            )

        # throughput-first: treat early, accept inspection-dependent extended
        throughput = build(JointPolicy(extended_holdover_penalty_min=0.0))
        self.assertEqual(throughput.tier, "extended")
        self.assertEqual(throughput.treatment_window.start, T0)
        # safety floor: extended margin plans are inadmissible
        safe = build(JointPolicy(min_margin_min=5))
        self.assertEqual(safe.tier, "conservative")
        self.assertGreaterEqual(safe.margin_min, 5)
        self.assertGreater(safe.treatment_window.start, T0)

    def test_no_feasible_window_when_holdover_cannot_reach_runway(self):
        # runway is fine, but the aircraft cannot even reach it before the
        # treatment's optimistic holdover expires (treatment+taxi > hmax)
        dm, rm = deicing_manager(), runway_manager()
        planner = DeicingDeparturePlanner(dm, rm)
        req = JointRequest(
            "JX1", "L", T0, T0 + timedelta(minutes=40), T0 + timedelta(hours=2),
            45,  # taxi alone exceeds the 30-minute optimistic holdover
            36.0, "T4", 100, "snow", -5.0, 100.0,
        )
        with self.assertRaises(RuntimeError):
            planner.plan(req)

    def test_incompatible_pad_is_rejected(self):
        dm = DeicingManager(
            [Pad("NARROW", 1, frozenset({"T1"}), 10.0, 100.0)], BANDS
        )
        planner = DeicingDeparturePlanner(dm, runway_manager())
        req = JointRequest(
            "JX1", "L", T0, T0, T0 + timedelta(hours=2), 5,
            36.0, "T4", 100, "snow", -5.0, 100.0,
        )
        with self.assertRaises(RuntimeError):
            planner.plan(req)


if __name__ == "__main__":
    unittest.main()
