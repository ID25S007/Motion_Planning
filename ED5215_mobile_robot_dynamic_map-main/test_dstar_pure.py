"""
test_dstar_pure.py
==================
Pure Python (no numpy, no matplotlib) test for D* Lite.
Verifies that D* / D* Lite also reaches all 8 goals.

Run:  python test_dstar_pure.py
Or in Spyder:  %run test_dstar_pure.py
"""

import heapq, random
from collections import deque

# ─── Config (mirrors grid_nav_viz.py exactly) ──────────────────────────────
GRID_ROWS   = 22
GRID_COLS   = 28
ROBOT_START = (1, 1)
MAX_STEPS   = 8000
SEED        = 99
COLLECT_DIST = 1
NUM_OBS     = 5
OBS_SPREAD  = 1
OBS_EVERY   = 12

GOALS = [
    (2,  25), (10, 13), (19,  2), (5,   7),
    (18, 24), (1,  20), (12,  4), (8,  22),
]

INNER_WALLS = [
    (3,3,2,12),(3,3,14,22),(7,7,6,14),(7,7,16,25),
    (11,11,2,8),(11,11,10,18),(15,15,5,13),(15,15,15,24),
    (19,19,3,9),(19,19,12,20),(1,5,14,14),(5,11,9,9),
    (8,15,21,21),(11,19,14,14),(3,9,19,19),(10,19,5,5),
    (4,12,24,24),
]

INF = float('inf')


# ─── Maze builder ──────────────────────────────────────────────────────────
def build_maze():
    g = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    for c in range(GRID_COLS): g[0][c] = g[GRID_ROWS-1][c] = 1
    for r in range(GRID_ROWS): g[r][0] = g[r][GRID_COLS-1] = 1
    for r1,r2,c1,c2 in INNER_WALLS:
        r1=max(1,min(GRID_ROWS-2,r1)); r2=max(1,min(GRID_ROWS-2,r2))
        c1=max(1,min(GRID_COLS-2,c1)); c2=max(1,min(GRID_COLS-2,c2))
        for r in range(r1,r2+1):
            for c in range(c1,c2+1): g[r][c]=1
    return g

def grid_copy(g): return [row[:] for row in g]

def nbs(r, c):
    for dr,dc in((-1,0),(1,0),(0,-1),(0,1)):
        nr,nc=r+dr,c+dc
        if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS: yield nr,nc


# ─── Pure Python D* Lite ───────────────────────────────────────────────────
class DStarLite:
    def __init__(self):
        self._g   = {}   # {cell: float}
        self._rhs = {}   # {cell: float}
        self.km = 0.0
        self.q  = []
        self.start = self.goal = self.grid = self.last_start = None

    def _g_(self,s): return self._g.get(s, INF)
    def _rhs_(self,s): return self._rhs.get(s, INF)
    def _h(self,a,b): return abs(a[0]-b[0])+abs(a[1]-b[1]) if a and b else 0.0
    def _key(self,s):
        v=min(self._g_(s),self._rhs_(s))
        return (v+self._h(s,self.start)+self.km, v)

    def _free(self,s):
        r,c=s
        if 0<=r<GRID_ROWS and 0<=c<GRID_COLS:
            return self.grid[r][c]==0 or s==self.start or s==self.goal
        return False

    def _update(self,u):
        if u!=self.goal:
            best=INF
            for nr,nc in nbs(*u):
                nb=(nr,nc)
                cost=1 if self._free(nb) else INF
                best=min(best,self._g_(nb)+cost)
            self._rhs[u]=best
        self.q=[x for x in self.q if x[1]!=u]
        if self._g_(u)!=self._rhs_(u):
            heapq.heappush(self.q,(self._key(u),u))

    def _compute(self):
        cap=0
        while self.q and (self.q[0][0]<self._key(self.start) or
                          self._rhs_(self.start)!=self._g_(self.start)):
            cap+=1
            if cap>200000: break
            k_old,u=heapq.heappop(self.q)
            if k_old<self._key(u):
                heapq.heappush(self.q,(self._key(u),u)); continue
            if self._g_(u)>self._rhs_(u):
                self._g[u]=self._rhs_(u)
                for nr,nc in nbs(*u): self._update((nr,nc))
            else:
                self._g[u]=INF; self._update(u)
                for nr,nc in nbs(*u): self._update((nr,nc))

    def _extract(self,start,goal):
        if start==goal: return [start]
        if self._g_(start)==INF: return []
        path=[start]; cur=start; seen={start}
        while cur!=goal:
            best,bv=None,INF
            for nr,nc in nbs(*cur):
                nb=(nr,nc)
                if self._free(nb) and nb not in seen and self._g_(nb)<bv:
                    bv=self._g_(nb); best=nb
            if best is None: return []
            path.append(best); seen.add(best); cur=best
        return path

    def plan(self, grid, start, goal):
        if self.grid is None or self.goal!=goal:
            # Full reset for new goal
            self._g={}; self._rhs={}; self.km=0.0; self.q=[]
            self.grid=grid_copy(grid); self.start=start
            self.last_start=start; self.goal=goal
            self._rhs[goal]=0
            heapq.heappush(self.q,(self._key(goal),goal))
        else:
            self.km+=self._h(self.last_start,start)
            self.last_start=start; self.start=start
            # detect changes
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    if self.grid[r][c]!=grid[r][c]:
                        self.grid[r][c]=grid[r][c]; u=(r,c)
                        self._update(u)
                        for nr,nc in nbs(*u): self._update((nr,nc))
        self._compute()
        return self._extract(start,goal)


# ─── A* (reference) ────────────────────────────────────────────────────────
def astar(grid,start,goal):
    if start==goal: return [start]
    h=lambda s:abs(s[0]-goal[0])+abs(s[1]-goal[1])
    q=[(h(start),0,start)]; gs={start:0}; cf={}; vis=set()
    while q:
        _,g,cur=heapq.heappop(q)
        if cur in vis: continue
        vis.add(cur)
        if cur==goal:
            p=[]; n=goal
            while n!=start: p.append(n); n=cf[n]
            p.append(start); p.reverse(); return p
        for nr,nc in nbs(*cur):
            nb=(nr,nc)
            if grid[nr][nc]!=0 and nb!=goal and nb!=start: continue
            ng=g+1
            if ng<gs.get(nb,INF):
                gs[nb]=ng; cf[nb]=cur
                heapq.heappush(q,(ng+h(nb),ng,nb))
    return []


# ─── Environment ───────────────────────────────────────────────────────────
class Env:
    def __init__(self,planner_fn,seed=SEED):
        random.seed(seed)
        self.base=build_maze()
        self.robot=list(ROBOT_START)
        self.goals=[g for g in GOALS if self.base[g[0]][g[1]]==0]
        self.goal_idx=0; self.collected=set()
        self.planned=[]; self.done=False; self.frame_n=0
        self.planner_fn=planner_fn; self.steps=0
        safety={tuple(self.robot)}|{tuple(g) for g in self.goals}
        self.dyn_obs=[]; tries=0
        while len(self.dyn_obs)<NUM_OBS and tries<2000:
            r=random.randint(1,GRID_ROWS-2); c=random.randint(1,GRID_COLS-2)
            if self.base[r][c]==0 and (r,c) not in safety:
                self.dyn_obs.append([r,c]); safety.add((r,c))
            tries+=1
        self._replan()

    def _inflate(self):
        cells=set()
        for o in self.dyn_obs:
            for dr in range(-OBS_SPREAD,OBS_SPREAD+1):
                for dc in range(-OBS_SPREAD,OBS_SPREAD+1):
                    nr,nc=o[0]+dr,o[1]+dc
                    if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS: cells.add((nr,nc))
        return cells

    def _pgrid(self):
        g=grid_copy(self.base)
        gc=tuple(self.goals[self.goal_idx]) if self.goal_idx<len(self.goals) else None
        rc=tuple(self.robot)
        for nr,nc in self._inflate():
            if (nr,nc) not in(rc,gc): g[nr][nc]=1
        return g

    def _move_obs(self):
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        occ={(o[0],o[1]) for o in self.dyn_obs}; rc=tuple(self.robot)
        for obs in self.dyn_obs:
            random.shuffle(dirs)
            for dr,dc in dirs:
                nr,nc=obs[0]+dr,obs[1]+dc
                if(1<=nr<GRID_ROWS-1 and 1<=nc<GRID_COLS-1
                   and self.base[nr][nc]==0
                   and(nr,nc) not in occ and(nr,nc)!=rc):
                    occ.discard((obs[0],obs[1])); obs[0],obs[1]=nr,nc; occ.add((nr,nc)); break

    def _advance(self):
        while self.goal_idx<len(self.goals) and self.goal_idx in self.collected:
            self.goal_idx+=1

    def _replan(self):
        self._advance()
        if self.goal_idx>=len(self.goals):
            uncol=[i for i in range(len(self.goals)) if i not in self.collected]
            if not uncol: self.planned=[]; self.done=True
            else: self.goal_idx=uncol[0]; self._replan()
            return
        curr=tuple(self.goals[self.goal_idx]); rob=tuple(self.robot)
        if abs(rob[0]-curr[0])+abs(rob[1]-curr[1])<=COLLECT_DIST:
            self.collected.add(self.goal_idx); self.goal_idx+=1; self._replan(); return
        path=self.planner_fn(self._pgrid(),rob,curr)
        self.planned=list(path[1:]) if len(path)>1 else []

    def step(self):
        if self.done: return
        self.frame_n+=1; self.steps+=1
        if self.frame_n%OBS_EVERY==0:
            self._move_obs()
            if self.planned:
                inf=self._inflate()
                if any(c in inf for c in self.planned[:4]): self._replan()
        if not self.planned: self._replan()
        if not self.planned: return
        nxt=self.planned.pop(0); self.robot=list(nxt); rob=tuple(self.robot)
        hit=False
        for i,g in enumerate(self.goals):
            if i not in self.collected:
                if abs(rob[0]-g[0])+abs(rob[1]-g[1])<=COLLECT_DIST:
                    self.collected.add(i); hit=True
        if hit: self.planned=[]; self._replan()


# ─── Run ───────────────────────────────────────────────────────────────────
# Two separate D* Lite instances (one for D*, one for D* Lite — same algo,
# different isolated instances to match the video generation setup)
dstar_inst1 = DStarLite()
dstar_inst2 = DStarLite()

ALGOS = [
    ('D*      ', lambda g,s,t: dstar_inst1.plan(g,s,t)),
    ('D* Lite ', lambda g,s,t: dstar_inst2.plan(g,s,t)),
]

bar = "=" * 62
print(bar)
print(f"  D* / D* LITE GOAL COLLECTION TEST")
print(f"  Grid={GRID_ROWS}x{GRID_COLS}  Goals={len(GOALS)}  MaxSteps={MAX_STEPS}  Seed={SEED}")
print(bar)

# Quick maze sanity check
base=build_maze()
ok=[g for g in GOALS if base[g[0]][g[1]]==0]
print(f"  Free goals: {len(ok)}/{len(GOALS)}", "✓" if len(ok)==len(GOALS) else "✗ WALL BLOCKED!")

# BFS reachability
vis=set(); q=deque([ROBOT_START]); vis.add(ROBOT_START)
while q:
    r,c=q.popleft()
    for nr,nc in nbs(r,c):
        if base[nr][nc]==0 and (nr,nc) not in vis: vis.add((nr,nc)); q.append((nr,nc))
unreachable=[g for g in GOALS if g not in vis]
print(f"  Unreachable goals: {unreachable if unreachable else 'none'}", "✓" if not unreachable else "✗")

print(f"\n  Simulation results:", flush=True)
all_pass=True
for name,fn in ALGOS:
    env=Env(fn)
    for _ in range(MAX_STEPS):
        env.step()
        if env.done: break
    n=len(env.collected); t=len(env.goals); ok2=n==t
    if not ok2: all_pass=False
    miss=[env.goals[i] for i in range(t) if i not in env.collected] if not ok2 else []
    tag="PASS ✓" if ok2 else "FAIL ✗"
    to=" (TIMED OUT)" if not env.done else ""
    print(f"    [{tag}] {name}  {n}/{t} goals  steps={env.steps}{to}")
    if miss: print(f"           Missing: {miss}")

print(bar)
print("  ✓ D* and D* Lite BOTH REACH ALL 8 GOALS\n" if all_pass
      else "  ✗ Some tests failed — see above\n")
