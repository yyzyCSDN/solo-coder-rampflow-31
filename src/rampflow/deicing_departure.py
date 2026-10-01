from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Iterable
from .deicing import DeicingManager,Treatment
from .runway_slots import RunwaySlotManager,RunwayUse,SlotRequest
from .timeutil import utc

@dataclass(frozen=True)
class DeparturePlan:
    """A committed deicing treatment paired with a runway departure slot.

    ``margin_min`` is the achieved safety margin: minutes between the
    departure slot and the conservative holdover expiry.  It is always
    greater than or equal to the ``min_margin_min`` the planner was asked
    for, and larger when traffic is light.
    """
    flight_id:str
    treatment:Treatment
    runway_use:RunwayUse
    margin_min:int
    @property
    def departure(self)->datetime:
        return self.runway_use.slot
    @property
    def expires_at(self)->datetime:
        return self.treatment.conservative_expiry

class DeicingDeparturePlanner:
    """Jointly books a deicing bay and a runway slot for one departure.

    A deicing treatment starts the holdover clock, so booking the bay and
    the runway independently either wastes holdover time waiting for a
    slot or accepts a departure minutes before expiry.  This planner
    searches both resources together: it takes the earliest deicing window
    whose paired runway slot still leaves at least ``min_margin_min`` of
    holdover in reserve.  Raising the margin trades throughput (later or
    fewer departures) for safety; setting it to zero maximises throughput.
    """

    def __init__(self,deicing:DeicingManager,runway:RunwaySlotManager,taxi_out_min:int=10,search_step_min:int=5,max_attempts:int=48):
        if taxi_out_min<0 or search_step_min<=0 or max_attempts<=0:
            raise ValueError("invalid planner parameters")
        self.deicing=deicing
        self.runway=runway
        self.taxi_out_min=taxi_out_min
        self.search_step_min=search_step_min
        self.max_attempts=max_attempts
        self.plans:dict[str,DeparturePlan]={}

    def plan(self,flight_id:str,pad_id:str,ready:datetime,wingspan_m:float,fluid_type:str,concentration:int,precip:str,temp_c:float,fluid_liters:float,wake_category:str,operation:str="departure",runway_preferences:Iterable[str]=(),min_margin_min:int=0,ctot:datetime|None=None,ctot_tolerance_min:int=5)->DeparturePlan:
        if flight_id in self.plans:
            raise ValueError("flight already planned")
        if min_margin_min<0:
            raise ValueError("negative safety margin")
        ready=utc(ready)
        hmin,_=self.deicing.holdover(precip,fluid_type,temp_c,concentration)
        if self.taxi_out_min+min_margin_min>hmin:
            raise ValueError("holdover too short for taxi-out plus safety margin")
        duration=self.deicing.estimate_minutes(pad_id,wingspan_m,fluid_liters)
        cursor=ready
        last_error:Exception|None=None
        for _ in range(self.max_attempts):
            slot=self.deicing.find_slot(pad_id,cursor,duration)
            expiry=slot.end+timedelta(minutes=hmin)
            earliest_dep=slot.end+timedelta(minutes=self.taxi_out_min)
            latest_dep=expiry-timedelta(minutes=min_margin_min)
            req=SlotRequest(flight_id,operation,wake_category,earliest_dep,earliest_dep,latest_dep,tuple(runway_preferences),ctot,ctot_tolerance_min)
            try:
                use=self.runway.find(req)
            except RuntimeError as exc:
                last_error=exc
            else:
                return self._commit(flight_id,pad_id,slot,wingspan_m,fluid_type,concentration,precip,temp_c,fluid_liters,use)
            cursor=slot.start+timedelta(minutes=self.search_step_min)
        raise RuntimeError("no joint deicing/departure window")from last_error

    def _commit(self,flight_id:str,pad_id:str,slot,wingspan_m:float,fluid_type:str,concentration:int,precip:str,temp_c:float,fluid_liters:float,use:RunwayUse)->DeparturePlan:
        treatment=self.deicing.schedule(flight_id,pad_id,slot.start,wingspan_m,fluid_type,concentration,precip,temp_c,fluid_liters)
        if treatment.start!=slot.start or treatment.end!=slot.end:
            self.deicing.cancel(flight_id)
            raise RuntimeError("deicing window moved while committing")
        decision=self.runway.commit(use)
        if not decision.accepted:
            self.deicing.cancel(flight_id)
            raise RuntimeError(f"runway slot rejected: {decision.reason}")
        margin=int((treatment.conservative_expiry-use.slot).total_seconds()//60)
        plan=DeparturePlan(flight_id,treatment,use,margin)
        self.plans[flight_id]=plan
        return plan

    def release_expired(self,now:datetime)->tuple[str,...]:
        """Release the bay and runway slot of every plan past its holdover.

        An expired plan must keep occupying nothing: the bay window is
        removed from the pad and the runway slot stops blocking later
        traffic.  Returns the released flight ids.
        """
        now=utc(now)
        expired=sorted(fid for fid,plan in self.plans.items() if now>plan.expires_at)
        for fid in expired:
            self._release(fid)
        return tuple(expired)

    def cancel(self,flight_id:str)->bool:
        if flight_id not in self.plans:
            return False
        self._release(flight_id)
        return True

    def _release(self,flight_id:str)->None:
        self.deicing.cancel(flight_id)
        self.runway.release(flight_id)
        del self.plans[flight_id]
