from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from typing import Iterable,Mapping
from .timeutil import Window,utc
from .models import Decision

@dataclass(frozen=True)
class Runway:
    runway_id:str
    directions:frozenset[str]
    max_tailwind_kt:float
    max_crosswind_kt:float
    occupancy_seconds:int
    mixed_mode:bool=True

@dataclass(frozen=True)
class SlotRequest:
    flight_id:str
    operation:str
    wake_category:str
    target:datetime
    earliest:datetime
    latest:datetime
    runway_preferences:tuple[str,...]=()
    ctot:datetime|None=None
    ctot_tolerance_min:int=5
    def __post_init__(self):
        e,t,l=utc(self.earliest),utc(self.target),utc(self.latest)
        if not e<=t<=l:raise ValueError("invalid slot window")
        object.__setattr__(self,"earliest",e);object.__setattr__(self,"target",t);object.__setattr__(self,"latest",l)
        if self.ctot is not None:object.__setattr__(self,"ctot",utc(self.ctot))

@dataclass(frozen=True)
class RunwayUse:
    flight_id:str
    runway_id:str
    operation:str
    wake_category:str
    slot:datetime
    occupied:Window
    sequence:int

class RunwaySlotManager:
    def __init__(self,runways:Iterable[Runway],wake_separation_seconds:Mapping[tuple[str,str],int],grid_seconds:int=30):
        self.runways={r.runway_id:r for r in runways}
        self.wake=dict(wake_separation_seconds)
        self.grid_seconds=grid_seconds
        self._uses:dict[str,list[RunwayUse]]={r:[] for r in self.runways}
        self._sequence=0
        self._config:frozenset[str]=frozenset(self.runways)

    def configure(self,active_runways:Iterable[str])->None:
        active=frozenset(active_runways)
        if not active or not active.issubset(self.runways):raise ValueError("invalid runway configuration")
        self._config=active

    def _quantize(self,t:datetime,ceil:bool=True)->datetime:
        t=utc(t)
        epoch=int(t.timestamp())
        q=self.grid_seconds
        if ceil:
            rounded=((epoch+q-1)//q)*q
        else:
            rounded=(epoch//q)*q
        return datetime.fromtimestamp(rounded,tz=t.tzinfo)

    def _sep(self, leader: str, follower: str) -> int:
        # Wake separation is directional: the pair key is (leader, follower).
        # It must never be key-sorted, which swaps leader and follower and can
        # hand back an under-sized gap (e.g. light leader / heavy follower).
        direct = self.wake.get((leader, follower))
        if direct is not None:
            return int(direct)
        follower_default = self.wake.get(("*", follower))
        if follower_default is not None:
            return int(follower_default)
        reverse = self.wake.get((follower, leader))
        if reverse is not None:
            # Last-resort backstop for incomplete tables: using the reverse
            # pairing is non-standard, so callers should supply follower
            # defaults ("*", follower) rather than rely on this.
            return int(reverse)
        return int(self.wake.get(("*", leader), 60))

    def _ctot_ok(self,req:SlotRequest,slot:datetime)->bool:
        if req.ctot is None:return True
        tol=timedelta(minutes=req.ctot_tolerance_min)
        return req.ctot-tol<=slot<=req.ctot+tol

    def _conflict(self,runway:Runway,req:SlotRequest,slot:datetime)->RunwayUse|None:
        occupancy=Window(slot,slot+timedelta(seconds=runway.occupancy_seconds))
        uses=self._uses[runway.runway_id]
        for use in uses:
            if use.occupied.overlaps(occupancy):
                return use
            if use.slot<=slot:
                sep=self._sep(use.wake_category,req.wake_category)
                if slot<use.slot+timedelta(seconds=sep):return use
            else:
                sep=self._sep(req.wake_category,use.wake_category)
                if use.slot<slot+timedelta(seconds=sep):return use
            if not runway.mixed_mode and use.operation!=req.operation:
                if abs((use.slot-slot).total_seconds())<max(90,runway.occupancy_seconds):
                    return use
        return None

    def _wind_ok(self,runway:Runway,tailwind_kt:float,crosswind_kt:float)->bool:
        return tailwind_kt<=runway.max_tailwind_kt and crosswind_kt<=runway.max_crosswind_kt

    def conflict_for(self,runway_id:str,operation:str,wake_category:str,slot:datetime,occupancy_seconds:int|None=None,ignore_flight:str|None=None)->RunwayUse|None:
        """Read-only conflict probe against committed traffic, optionally
        ignoring one flight (used when a plan re-books its own release)."""
        runway=self.runways[runway_id]
        occ=occupancy_seconds if occupancy_seconds is not None else runway.occupancy_seconds
        probe=Window(slot,slot+timedelta(seconds=occ))
        for use in self._uses[runway_id]:
            if ignore_flight is not None and use.flight_id==ignore_flight:continue
            if use.occupied.overlaps(probe):
                return use
            if use.slot<=slot:
                if slot<use.slot+timedelta(seconds=self._sep(use.wake_category,wake_category)):return use
            else:
                if use.slot<slot+timedelta(seconds=self._sep(wake_category,use.wake_category)):return use
            if not runway.mixed_mode and use.operation!=operation:
                if abs((use.slot-slot).total_seconds())<max(90,runway.occupancy_seconds):
                    return use
        return None

    def find(self,req:SlotRequest,wind_by_runway:Mapping[str,tuple[float,float]]|None=None)->RunwayUse:
        wind=wind_by_runway or {}
        runways=[r for r in req.runway_preferences if r in self._config] or sorted(self._config)
        options=[]
        for rid in runways:
            runway=self.runways[rid]
            tail,cross=wind.get(rid,(0.0,0.0))
            if not self._wind_ok(runway,tail,cross):continue
            cursor=self._quantize(req.earliest,True)
            while cursor<=req.latest:
                if self._ctot_ok(req,cursor):
                    hit=self._conflict(runway,req,cursor)
                    if hit is None:
                        distance=abs((cursor-req.target).total_seconds())
                        options.append((distance,cursor,rid,RunwayUse(req.flight_id,rid,req.operation,req.wake_category,cursor,Window(cursor,cursor+timedelta(seconds=runway.occupancy_seconds)),self._sequence+1)))
                        break
                    cursor=self._quantize(max(cursor+timedelta(seconds=self.grid_seconds),hit.occupied.end),True)
                else:
                    if req.ctot and cursor<req.ctot-timedelta(minutes=req.ctot_tolerance_min):
                        cursor=self._quantize(req.ctot-timedelta(minutes=req.ctot_tolerance_min),True)
                    else:
                        cursor+=timedelta(seconds=self.grid_seconds)
        if not options:raise RuntimeError("no runway slot")
        options.sort(key=lambda x:(x[0],x[1],x[2]))
        return options[0][3]

    def commit(self,use:RunwayUse)->Decision:
        if use.runway_id not in self._config:
            return Decision(False,"configuration","runway no longer active")
        req=SlotRequest(use.flight_id,use.operation,use.wake_category,use.slot,use.slot,use.slot,(use.runway_id,))
        hit=self._conflict(self.runways[use.runway_id],req,use.slot)
        if hit is not None:
            return Decision(False,"slot-stale","runway sequence changed",{"conflict":hit.flight_id})
        self._sequence+=1
        committed=RunwayUse(use.flight_id,use.runway_id,use.operation,use.wake_category,use.slot,use.occupied,self._sequence)
        self._uses[use.runway_id].append(committed)
        self._uses[use.runway_id].sort(key=lambda x:(x.slot,x.sequence))
        return Decision(True,"ok","runway slot committed",{"sequence":self._sequence})

    def resequence(self,runway_id:str,flight_order:Iterable[str],anchor:datetime)->tuple[RunwayUse,...]:
        order=list(flight_order)
        existing={u.flight_id:u for u in self._uses[runway_id] if u.flight_id in order}
        if set(existing)!=set(order):raise ValueError("flight order does not match active uses")
        runway=self.runways[runway_id]
        cursor=self._quantize(anchor,True)
        updated=[]
        leader=None
        for flight_id in order:
            old=existing[flight_id]
            if leader is not None:
                cursor=max(cursor,leader.slot+timedelta(seconds=self._sep(leader.wake_category,old.wake_category)))
                cursor=self._quantize(cursor,True)
            item=RunwayUse(old.flight_id,old.runway_id,old.operation,old.wake_category,cursor,Window(cursor,cursor+timedelta(seconds=runway.occupancy_seconds)),old.sequence)
            updated.append(item);leader=item
        other=[u for u in self._uses[runway_id] if u.flight_id not in existing]
        for item in updated:
            if any(o.occupied.overlaps(item.occupied) for o in other):raise RuntimeError("resequence collides with fixed traffic")
        self._uses[runway_id]=sorted(other+updated,key=lambda x:(x.slot,x.sequence))
        return tuple(updated)

    def release(self,flight_id:str)->bool:
        """Give back a committed time slot so the runway can serve other
        traffic. Expired or cancelled plans must not keep holding slots."""
        removed=False
        for rid in list(self._uses):
            kept=[u for u in self._uses[rid] if u.flight_id!=flight_id]
            if len(kept)!=len(self._uses[rid]):
                removed=True
                self._uses[rid]=kept
        return removed
