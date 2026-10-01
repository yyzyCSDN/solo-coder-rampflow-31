from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from decimal import Decimal, ROUND_FLOOR
from typing import Iterable
from .timeutil import Window,utc
from .models import Decision

@dataclass(frozen=True)
class HoldoverBand:
    precip: str
    fluid_type: str
    min_temp_c: float
    max_temp_c: float
    min_minutes: int
    max_minutes: int
    concentration: int = 100
    def applies(self,precip:str,fluid_type:str,temp_c:float,concentration:int)->bool:
        return self.precip==precip and self.fluid_type==fluid_type and self.min_temp_c<=temp_c<self.max_temp_c and concentration>=self.concentration

@dataclass(frozen=True)
class Treatment:
    flight_id: str
    pad_id: str
    bay: int
    start: datetime
    end: datetime
    fluid_type: str
    concentration: int
    precip: str
    temp_c: float
    holdover_min: int
    holdover_max: int
    inspection_interval_min: int
    sequence: int
    @property
    def conservative_expiry(self)->datetime:
        return self.end+timedelta(minutes=self.holdover_min)
    @property
    def optimistic_expiry(self)->datetime:
        return self.end+timedelta(minutes=self.holdover_max)

@dataclass(frozen=True)
class Pad:
    pad_id:str
    bays:int
    supported_fluids:frozenset[str]
    max_wingspan_m:float
    throughput_lpm:float
    def __post_init__(self):
        if self.bays<1:
            raise ValueError("pad needs at least one bay")

class DeicingManager:
    def __init__(self,pads:Iterable[Pad],bands:Iterable[HoldoverBand]):
        self.pads={p.pad_id:p for p in pads}
        self.bands=tuple(bands)
        self.treatments:dict[str,Treatment]={}
        # one independent occupancy timeline per bay; bays operate in parallel
        self._pad_use:dict[str,list[list[Window]]]={
            p.pad_id:[[] for _ in range(p.bays)] for p in self.pads.values()
        }
        self._sequence=0

    def _holdover(self,precip:str,fluid_type:str,temp_c:float,concentration:int)->tuple[int,int]:
        candidates=[b for b in self.bands if b.applies(precip,fluid_type,temp_c,concentration)]
        if not candidates:
            raise ValueError("no holdover guidance")
        candidates.sort(key=lambda b:(b.concentration,b.min_temp_c),reverse=True)
        chosen=candidates[0]
        return chosen.min_minutes,chosen.max_minutes

    def _inspection_interval(self,holdover_min:int,precip:str)->int:
        factor={"freezing-rain":Decimal("0.25"),"heavy-snow":Decimal("0.33"),"snow":Decimal("0.50"),"frost":Decimal("0.75")}.get(precip,Decimal("0.50"))
        return max(2,int((Decimal(holdover_min)*factor).quantize(Decimal("1"),rounding=ROUND_FLOOR)))

    def estimate_minutes(self,pad_id:str,wingspan_m:float,fluid_liters:float)->int:
        pad=self.pads[pad_id]
        if wingspan_m>pad.max_wingspan_m:
            raise ValueError("aircraft too large for pad")
        base=fluid_liters/max(1.0,pad.throughput_lpm)
        setup=4.0 + max(0.0,wingspan_m-36.0)/12.0
        return max(5,int(base+setup+0.999))

    def earliest_window(self,pad_id:str,earliest:datetime,duration_min:int)->Window:
        """Earliest non-overlapping window across every bay of the pad.

        Each bay keeps its own timeline, so a two-bay pad can serve two
        aircraft at the same instant instead of queueing the second one.
        """
        return self.earliest_bay_window(pad_id,earliest,duration_min)[1]

    def earliest_bay_window(self,pad_id:str,earliest:datetime,duration_min:int)->tuple[int,Window]:
        """Like earliest_window, but also reports which bay serves it."""
        start=utc(earliest)
        duration=timedelta(minutes=duration_min)
        best:Window|None=None
        best_bay=-1
        for bay in range(self.pads[pad_id].bays):
            cursor=start
            for _ in range(300):
                candidate=Window(cursor,cursor+duration)
                overlaps=[u for u in self._pad_use[pad_id][bay] if u.overlaps(candidate)]
                if not overlaps:
                    # equal earliest start across bays: keep the lowest bay
                    # index so a free second bay is used, not a later slot
                    if best is None or candidate.start<best.start or (candidate.start==best.start and bay<best_bay):
                        best=candidate;best_bay=bay
                    break
                cursor=max(u.end for u in overlaps)
        if best is None:
            raise RuntimeError("no deicing slot")
        return best_bay,best

    def latest_bay_window(self,pad_id:str,ready_by:datetime,latest_start:datetime,duration_min:int)->tuple[int,Window]|None:
        """Like latest_window, but also reports which bay serves it.

        Computed per bay as the latest free interval of length duration whose
        start lies within [ready_by, latest_start], then taking the latest
        across bays (lowest bay index on ties).
        """
        ready=utc(ready_by); cap=utc(latest_start)
        duration=timedelta(minutes=duration_min)
        best:Window|None=None
        best_bay=-1
        for bay in range(self.pads[pad_id].bays):
            cursor=ready
            candidate:Window|None=None
            for used in sorted(self._pad_use[pad_id][bay],key=lambda w:w.start):
                if used.start>=cursor+duration:
                    candidate=Window(min(cap,used.start-duration),min(cap,used.start-duration)+duration)
                cursor=max(cursor,used.end)
                if cursor>cap:
                    break
            if cursor<=cap:
                candidate=Window(cap,cap+duration)
            if candidate is not None and candidate.start>=ready and candidate.start<=cap:
                if best is None or candidate.start>best.start or (candidate.start==best.start and bay<best_bay):
                    best=candidate;best_bay=bay
        if best is None:
            return None
        return best_bay,best

    def latest_window(self,pad_id:str,ready_by:datetime,latest_start:datetime,duration_min:int)->Window|None:
        """Latest window that can begin within [ready_by, latest_start].

        Pushing treatment as late as possible keeps the holdover alive for a
        later departure without touching throughput (other traffic can still
        take the early slots).  Returns None when no bay can fit the job.
        """
        found=self.latest_bay_window(pad_id,ready_by,latest_start,duration_min)
        return None if found is None else found[1]

    def find_slot(self, pad_id: str, earliest: datetime, duration_min: int) -> Window:
        return self.earliest_window(pad_id,earliest,duration_min)

    def holdover_range(self,fluid_type:str,concentration:int,precip:str,temp_c:float)->tuple[int,int]:
        return self._holdover(precip,fluid_type,temp_c,concentration)

    def reserve(self,flight_id:str,pad_id:str,window:Window,wingspan_m:float,fluid_type:str,concentration:int,precip:str,temp_c:float,bay:int|None=None)->Treatment:
        """Book a specific pre-computed window, optionally on a pinned bay.

        Joint planning resolves the bay at search time so the booking cannot
        land on a different bay than the one the window was computed for.
        """
        if flight_id in self.treatments:
            raise ValueError("flight already treated")
        pad=self.pads[pad_id]
        if wingspan_m>pad.max_wingspan_m:
            raise ValueError("aircraft too large for pad")
        if fluid_type not in pad.supported_fluids:
            raise ValueError("fluid unsupported by pad")
        target=Window(utc(window.start),utc(window.end))
        if bay is not None:
            if not 0<=bay<pad.bays:
                raise ValueError("unknown bay")
            if any(u.overlaps(target) for u in self._pad_use[pad_id][bay]):
                raise RuntimeError("deicing bay conflict")
        else:
            bay=next((i for i,line in enumerate(self._pad_use[pad_id])
                      if not any(u.overlaps(target) for u in line)),None)
            if bay is None:
                raise RuntimeError("deicing bay conflict")
        hmin,hmax=self._holdover(precip,fluid_type,temp_c,concentration)
        self._sequence += 1
        treatment=Treatment(flight_id,pad_id,bay,target.start,target.end,fluid_type,concentration,precip,temp_c,hmin,hmax,self._inspection_interval(hmin,precip),self._sequence)
        self.treatments[flight_id]=treatment
        self._pad_use[pad_id][bay].append(target)
        self._pad_use[pad_id][bay].sort(key=lambda w:w.start)
        return treatment

    def schedule(self,flight_id:str,pad_id:str,earliest:datetime,wingspan_m:float,fluid_type:str,concentration:int,precip:str,temp_c:float,fluid_liters:float)->Treatment:
        if flight_id in self.treatments:
            raise ValueError("flight already treated")
        pad=self.pads[pad_id]
        if fluid_type not in pad.supported_fluids:
            raise ValueError("fluid unsupported by pad")
        hmin,hmax=self._holdover(precip,fluid_type,temp_c,concentration)
        duration=self.estimate_minutes(pad_id,wingspan_m,fluid_liters)
        slot=self.find_slot(pad_id,earliest,duration)
        self._sequence += 1
        bay=next(i for i,line in enumerate(self._pad_use[pad_id])
                 if not any(u.overlaps(slot) for u in line))
        treatment=Treatment(flight_id,pad_id,bay,slot.start,slot.end,fluid_type,concentration,precip,temp_c,hmin,hmax,self._inspection_interval(hmin,precip),self._sequence)
        self.treatments[flight_id]=treatment
        self._pad_use[pad_id][bay].append(slot)
        self._pad_use[pad_id][bay].sort(key=lambda w:w.start)
        return treatment

    def departure_status(self,flight_id:str,departure:datetime,last_inspection:datetime|None=None)->Decision:
        if flight_id not in self.treatments:
            return Decision(False,"untreated","no deicing treatment")
        t=self.treatments[flight_id]
        dep=utc(departure)
        if dep <= t.conservative_expiry:
            return Decision(True,"within-holdover","departure within conservative holdover",{"expiry":t.conservative_expiry})
        if dep > t.optimistic_expiry:
            return Decision(False,"expired","holdover guidance expired",{"expiry":t.optimistic_expiry})
        if last_inspection is None:
            return Decision(False,"inspection-required","departure is in extended holdover interval")
        inspected=utc(last_inspection)
        if inspected < t.end or inspected > dep:
            return Decision(False,"invalid-inspection","inspection timestamp outside valid interval")
        if dep-inspected > timedelta(minutes=t.inspection_interval_min):
            return Decision(False,"inspection-stale","post-treatment inspection is too old")
        return Decision(True,"inspected","extended holdover accepted after inspection",{"inspection":inspected})

    def cancel(self,flight_id:str)->bool:
        t=self.treatments.pop(flight_id,None)
        if t is None:return False
        window=Window(t.start,t.end)
        self._pad_use[t.pad_id][t.bay]=[x for x in self._pad_use[t.pad_id][t.bay] if x!=window]
        return True
