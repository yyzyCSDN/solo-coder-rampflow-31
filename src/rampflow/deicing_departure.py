"""Joint deicing and departure-window coordination.

Deicing only earns its keep if the aircraft can actually depart inside the
holdover window, so the two searches cannot be done independently:

* treating early frees the deicing bay for the queue (pad throughput) but the
  conservative holdover can expire before the allocated runway slot;
* treating as late as the runway slot permits keeps the holdover fresh at
  take-off (safety margin) but occupies the bay longer, and early treatment
  that slips past conservative expiry puts the flight into the
  inspection-dependent extended interval;
* taking a later runway slot adds departure delay but can be the only way to
  keep the departure inside holdover when the runway is saturated.

This planner enumerates feasible runway slots around both the earliest
possible departure and the requested target, pairs each with the earliest and
latest feasible treatment window on every bay of every usable pad, and scores
every combination with an explicit, tunable policy.  The winner is booked
atomically: a plan holds exactly one deicing bay and one runway time slot.
Cancellation, confirmed holdover expiry, a missed runway window, or a proposed
plan that was never worked all release both resources, so no dead plan keeps a
bay or a time slot occupied.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from enum import Enum
from typing import Mapping

from .deicing import DeicingManager
from .runway_slots import RunwaySlotManager,SlotRequest,RunwayUse
from .timeutil import Window,utc


class PlanState(str,Enum):
    PROPOSED="proposed"      # holding a bay + a runway slot, awaiting treatment
    CONFIRMED="confirmed"    # treatment delivered, waiting to depart
    DEPARTED="departed"
    CANCELLED="cancelled"
    EXPIRED="expired"        # window lapsed before departure; resources freed


@dataclass(frozen=True)
class JointPolicy:
    """How to trade throughput against holdover safety margin.

    All terms are denominated in minutes and summed into one score:

    * runway delay vs the requested target (runway throughput);
    * holdover margin at take-off, counted negatively (safety);
    * a fixed penalty for an extended-holdover / inspection-dependent plan.

    Pad throughput is not a third lever: the pad's own availability bounds the
    "latest treatment" candidate, so a treatment cannot be deferred into time a
    queued aircraft needs. margin_weight states what one extra minute of
    holdover safety is worth in minutes of pad/runway cost; plans below
    min_margin_min are inadmissible instead of merely costly.
    """
    delay_weight:float=1.0
    margin_weight:float=1.0
    extended_holdover_penalty_min:float=10.0
    min_margin_min:int=0
    prefer_pad:tuple[str,...]=()


@dataclass(frozen=True)
class JointRequest:
    flight_id:str
    wake_category:str
    ready:datetime                 # earliest the aircraft can reach a pad
    target_departure:datetime
    latest_departure:datetime
    taxi_minutes:int
    wingspan_m:float
    fluid_type:str
    concentration:int
    precip:str
    temp_c:float
    fluid_liters:float
    pad_preferences:tuple[str,...]=()
    runway_preferences:tuple[str,...]=()
    ctot:datetime|None=None
    ctot_tolerance_min:int=5
    def __post_init__(self):
        ready,target,latest=utc(self.ready),utc(self.target_departure),utc(self.latest_departure)
        if not ready<=target<=latest:raise ValueError("invalid deicing/departure window")
        if self.taxi_minutes<0:raise ValueError("negative taxi time")
        object.__setattr__(self,"ready",ready)
        object.__setattr__(self,"target_departure",target)
        object.__setattr__(self,"latest_departure",latest)
        if self.ctot is not None:object.__setattr__(self,"ctot",utc(self.ctot))


@dataclass(frozen=True)
class JointPlan:
    flight_id:str
    pad_id:str
    bay:int
    treatment_window:Window
    runway_id:str
    runway_slot:datetime
    holdover_min:int
    holdover_max:int
    tier:str                        # "conservative" or "extended"
    margin_min:float
    delay_min:float
    score:float
    state:PlanState
    expires_at:datetime             # end of the usable departure window
    treatment:Treatment|None=None
    runway_use:RunwayUse|None=None
    inspected:bool=False


class DeicingDeparturePlanner:
    def __init__(self,deicing:DeicingManager,runways:RunwaySlotManager,
                 policy:JointPolicy|None=None,
                 wind_by_runway:Mapping[str,tuple[float,float]]|None=None):
        self.deicing=deicing
        self.runways=runways
        self.policy=policy or JointPolicy()
        self.wind=dict(wind_by_runway or {})
        self.plans:dict[str,JointPlan]={}
        self._reqs:dict[str,JointRequest]={}

    # ---------------------------------------------------------- enumeration
    def _usable_pads(self,req:JointRequest)->list[str]:
        order=list(req.pad_preferences)+[p for p in self.deicing.pads if p not in req.pad_preferences]
        return [pid for pid in order
                if (pad:=self.deicing.pads.get(pid)) is not None
                and req.fluid_type in pad.supported_fluids
                and req.wingspan_m<=pad.max_wingspan_m]

    def _treatment_windows(self,req:JointRequest,pad_id:str,duration:int,
                           slot:datetime,hmin:int,hmax:int)->list[tuple[int,Window,str,datetime]]:
        """The two ends of the throughput/safety trade for a fixed runway slot.

        throughput: the earliest treatment each bay can serve; frees the bay
                    for the queue as soon as possible, but the conservative
                    holdover may already be lapsed at take-off (extended tier,
                    needs a post-treatment inspection).
        safety:     the latest treatment that still reaches the runway; keeps
                    the holdover freshest at take-off, at the cost of holding
                    the bay longer.
        Returns (bay, window, tier, expiry); treatments whose optimistic
        expiry precedes the slot are omitted because the deice would be dead.
        """
        latest_start_cap=slot-timedelta(minutes=req.taxi_minutes+duration)
        if latest_start_cap<req.ready:
            return []
        out:list[tuple[int,Window,str,datetime]]=[]
        candidates:list[tuple[int,Window]]=[]
        early=self.deicing.earliest_bay_window(pad_id,req.ready,duration)
        candidates.append(early)
        late=self.deicing.latest_bay_window(pad_id,req.ready,latest_start_cap,duration)
        if late is not None and (late[0],late[1].start)!=(early[0],early[1].start):
            candidates.append(late)
        for bay,window in candidates:
            if window.end>slot-timedelta(minutes=req.taxi_minutes):
                continue
            conservative=window.end+timedelta(minutes=hmin)
            optimistic=window.end+timedelta(minutes=hmax)
            if slot<=conservative:
                out.append((bay,window,"conservative",conservative))
            elif slot<=optimistic:
                out.append((bay,window,"extended",optimistic))
        return out

    def _runway_candidates(self,req:JointRequest,min_service_min:int=0,
                           window_min:int=30)->list[tuple[str,datetime]]:
        """Conflict-free runway slots per usable runway around two anchors:

        * the earliest physically possible departure (ready + treatment+taxi);
        * the requested target departure.

        A bounded window after each anchor is scanned so the trade between an
        early slot and a target-time slot (with fresher holdover) is visible,
        but enumeration cost stays bounded regardless of the planning horizon.
        """
        earliest=req.ready+timedelta(minutes=min_service_min)
        anchors=sorted({earliest,max(earliest,req.target_departure)})
        found:set[tuple[str,datetime]]=set()
        runways=[r for r in req.runway_preferences if r in self.runways._config] or sorted(self.runways._config)
        step=timedelta(seconds=self.runways.grid_seconds)
        span=timedelta(minutes=window_min)
        for rid in runways:
            runway=self.runways.runways[rid]
            tail,cross=self.wind.get(rid,(0.0,0.0))
            if not self.runways._wind_ok(runway,tail,cross):continue
            probe=SlotRequest(req.flight_id,"dep",req.wake_category,
                              max(req.target_departure,earliest),earliest,req.latest_departure,
                              (rid,),req.ctot,req.ctot_tolerance_min)
            for anchor in anchors:
                cursor=self.runways._quantize(anchor,True)
                deadline=min(req.latest_departure,cursor+span)
                while cursor<=deadline:
                    if self.runways._ctot_ok(probe,cursor) and self.runways._conflict(runway,probe,cursor) is None:
                        found.add((rid,cursor))
                    cursor+=step
        return sorted(found,key=lambda x:(x[1],x[0]))

    def _score(self,req:JointRequest,slot:datetime,
               window:Window,expiry:datetime,tier:str)->tuple[float,float,float]:
        delay=max(0.0,(slot-req.target_departure).total_seconds()/60.0)
        margin=(expiry-slot).total_seconds()/60.0
        penalty=self.policy.extended_holdover_penalty_min if tier=="extended" else 0.0
        score=(self.policy.delay_weight*delay
               -self.policy.margin_weight*margin
               +penalty)
        return score,delay,margin

    def plan(self,req:JointRequest)->JointPlan:
        if req.flight_id in self.plans:
            raise ValueError("flight already has an active plan")
        self._reqs[req.flight_id]=req
        try:
            plan=self._build_plan(req)
        except Exception:
            self._reqs.pop(req.flight_id,None)
            raise
        self.plans[req.flight_id]=plan
        return plan

    def _build_plan(self,req:JointRequest)->JointPlan:
        pads=self._usable_pads(req)
        if not pads:
            raise RuntimeError("no pad can handle this aircraft")
        pad_specs=[]
        min_service=10**9
        for pad_id in pads:
            hmin,hmax=self.deicing.holdover_range(req.fluid_type,req.concentration,req.precip,req.temp_c)
            duration=self.deicing.estimate_minutes(pad_id,req.wingspan_m,req.fluid_liters)
            pad_specs.append((pad_id,duration,hmin,hmax))
            min_service=min(min_service,duration+req.taxi_minutes)
        runway_slots=self._runway_candidates(req,min_service)
        if not runway_slots:
            raise RuntimeError("no departure window within runway capacity")
        options:list[tuple[float,datetime,str,str,int,datetime,JointPlan]]=[]
        for pad_id,duration,hmin,hmax in pad_specs:
            for rid,slot in runway_slots:
                for bay,window,tier,expiry in self._treatment_windows(req,pad_id,duration,slot,hmin,hmax):
                    score,delay,margin=self._score(req,slot,window,expiry,tier)
                    if margin<self.policy.min_margin_min:
                        continue
                    runway=self.runways.runways[rid]
                    use=RunwayUse(req.flight_id,rid,"dep",req.wake_category,slot,
                                  Window(slot,slot+timedelta(seconds=runway.occupancy_seconds)),
                                  self.runways._sequence+1)
                    plan=JointPlan(req.flight_id,pad_id,bay,window,rid,slot,
                                   hmin,hmax,tier,margin,delay,score,PlanState.PROPOSED,
                                   window.end+timedelta(minutes=hmax),None,use)
                    # ties favour the earlier slot, then the earlier treatment
                    # (frees the bay sooner) and lower pad/bay indices
                    options.append((score,slot,rid,pad_id,bay,window.start,plan))
        if not options:
            raise RuntimeError("no joint deicing/departure window")
        options.sort(key=lambda x:(x[0],x[1],x[2],x[3],x[5],x[4]))
        return options[0][6]

    # ------------------------------------------------------------- booking
    def book(self,plan:JointPlan)->JointPlan:
        """Atomically turn a plan into a held bay and held runway slot."""
        current=self.plans.get(plan.flight_id)
        if current is None or current.state!=PlanState.PROPOSED or current!=plan:
            raise ValueError("plan is not bookable")
        req=self._reqs[plan.flight_id]
        for _ in range(3):
            # reserve the pinned deicing bay first ...
            try:
                treatment=self.deicing.reserve(
                    plan.flight_id,plan.pad_id,plan.treatment_window,
                    req.wingspan_m,req.fluid_type,req.concentration,req.precip,req.temp_c,
                    bay=plan.bay)
            except (RuntimeError,ValueError):
                # pad changed underneath us; replan and retry
                plan=self._replan(plan)
                req=self._reqs[plan.flight_id]
                continue
            # ... then commit the runway slot, rolling the bay back on loss
            if self.runways.commit(plan.runway_use).accepted:
                booked=replace(plan,treatment=treatment)
                self.plans[plan.flight_id]=booked
                return booked
            self.deicing.cancel(plan.flight_id)
            plan=self._replan(plan)
            req=self._reqs[plan.flight_id]
        raise RuntimeError("joint booking failed under contention")

    def _replan(self,stale:JointPlan)->JointPlan:
        req=self._reqs[stale.flight_id]
        self.plans.pop(stale.flight_id,None)
        self.deicing.cancel(stale.flight_id)
        self.runways.release(stale.flight_id)
        plan=self._build_plan(req)
        self.plans[req.flight_id]=plan
        return plan

    # ---------------------------------------------------------- lifecycle
    def confirm_treatment(self,flight_id:str,at:datetime,inspection_passed:bool=False)->JointPlan:
        """Mark the deicing treatment delivered inside the booked window."""
        plan=self.plans[flight_id]
        if plan.state!=PlanState.PROPOSED:
            raise RuntimeError("plan not awaiting treatment")
        when=utc(at)
        if not plan.treatment_window.contains(when,end_inclusive=True):
            raise RuntimeError("treatment confirmation outside booked window")
        if when>plan.expires_at:
            self._release(flight_id,PlanState.EXPIRED)
            raise RuntimeError("holdover already expired at treatment time")
        return self._replace(plan,state=PlanState.CONFIRMED,inspected=inspection_passed)

    def record_inspection(self,flight_id:str)->JointPlan:
        plan=self.plans[flight_id]
        return self._replace(plan,inspected=True)

    def depart(self,flight_id:str,at:datetime)->JointPlan:
        plan=self.plans[flight_id]
        if plan.state==PlanState.DEPARTED:
            return plan
        if plan.state not in (PlanState.PROPOSED,PlanState.CONFIRMED):
            raise RuntimeError("plan no longer active")
        when=utc(at)
        if when>plan.expires_at:
            self._release(flight_id,PlanState.EXPIRED)
            raise RuntimeError("holdover expired before departure; bay and slot released")
        if plan.tier=="extended" and not plan.inspected:
            raise RuntimeError("extended-holdover departure needs a passing inspection")
        departed=self._replace(plan,state=PlanState.DEPARTED)
        # off-block: the bay is free; the runway slot is consumed by movement
        self.deicing.cancel(flight_id)
        return departed

    def cancel(self,flight_id:str)->bool:
        plan=self.plans.get(flight_id)
        if plan is None or plan.state in (PlanState.DEPARTED,PlanState.CANCELLED,PlanState.EXPIRED):
            return False
        return self._release(flight_id,PlanState.CANCELLED)

    def purge_expired(self,at:datetime)->tuple[str,...]:
        """Release plans that can no longer be flown.

        * proposed plans never confirmed once their treatment window starts
          slipping (the pad must clear for the next booking);
        * confirmed plans whose holdover has lapsed before departure.
        Bays and runway slots return to their pools.
        """
        when=utc(at); freed=[]
        for flight_id,plan in list(self.plans.items()):
            if plan.state in (PlanState.DEPARTED,PlanState.CANCELLED,PlanState.EXPIRED):
                continue
            deadline=(plan.expires_at if plan.state==PlanState.CONFIRMED
                      else plan.treatment_window.start)
            if when>deadline:
                self._release(flight_id,PlanState.EXPIRED)
                freed.append(flight_id)
        return tuple(freed)

    def _release(self,flight_id:str,state:PlanState)->bool:
        plan=self.plans[flight_id]
        self.deicing.cancel(flight_id)
        self.runways.release(flight_id)
        self.plans[flight_id]=replace(plan,state=state,treatment=None,runway_use=None)
        return True

    def _replace(self,plan:JointPlan,**changes)->JointPlan:
        updated=replace(plan,**changes)
        self.plans[plan.flight_id]=updated
        return updated
