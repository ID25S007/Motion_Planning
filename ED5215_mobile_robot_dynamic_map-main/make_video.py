"""
make_video.py
=============
Generates nav_result.gif (animated GIF) of the robot navigating.
Runs headlessly — no display needed.
~200 frames  @ 12 fps  ≈ 17 seconds
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np, random, os, sys

# ─── add project dir to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_planner import plan, solve_tsp

# ─── settings ────────────────────────────────────────────────────────────
GRID_ROWS, GRID_COLS = 18, 22
ROBOT_START = (1, 1)
GOALS = [(2,19),(8,10),(15,2),(4,6),(14,19)]
ALGORITHM   = 'astar'
NUM_OBS     = 4
OBS_EVERY   = 3
FPS         = 12
N_FRAMES    = 200
OUT         = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'nav_result.gif')

INNER_WALLS = [
    (3,3,2,9),(3,3,11,18),(7,7,4,11),(7,7,13,19),
    (11,11,2,7),(11,11,9,15),(14,14,4,11),(14,14,13,19),
    (1,4,11,11),(4,10,7,7),(8,14,17,17),(11,16,11,11),
]
C_BG='#0d0d1a'; C_WALL='#2e4057'; C_VIS='#00b894'
C_PLAN='#e17055'; C_GOAL='#fdcb6e'; C_DONE='#55efc4'
C_OBS='#d63031'; C_ROB='#74b9ff'

# ─── maze ────────────────────────────────────────────────────────────────
def build_maze():
    m = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int8)
    m[0,:]=m[-1,:]=m[:,0]=m[:,-1]=1
    for r1,r2,c1,c2 in INNER_WALLS:
        m[max(1,r1):min(GRID_ROWS-1,r2)+1,
          max(1,c1):min(GRID_COLS-1,c2)+1]=1
    return m

# ─── env ─────────────────────────────────────────────────────────────────
class Env:
    def __init__(self):
        self.base  = build_maze()
        self.robot = list(ROBOT_START)
        self.goals = [g for g in GOALS if self.base[g[0],g[1]]==0]
        self.goals = solve_tsp(ROBOT_START, self.goals) or self.goals
        self.gi=0; self.col=set(); self.done=False
        self.vis=[tuple(self.robot)]; self.plan=[]; self.fn=0
        self.obs=self._spawn(); self._replan()

    def _spawn(self):
        s={tuple(self.robot)}|{tuple(g) for g in self.goals}
        obs,t=[],0
        while len(obs)<NUM_OBS and t<2000:
            r,c=random.randint(1,GRID_ROWS-2),random.randint(1,GRID_COLS-2)
            if self.base[r,c]==0 and (r,c) not in s:
                obs.append([r,c]); s.add((r,c))
            t+=1
        return obs

    def _inf(self):
        s=set()
        for o in self.obs:
            for dr in(-1,0,1):
                for dc in(-1,0,1):
                    nr,nc=o[0]+dr,o[1]+dc
                    if 0<=nr<GRID_ROWS and 0<=nc<GRID_COLS: s.add((nr,nc))
        return s

    def _pg(self):
        g=self.base.copy()
        gc=tuple(self.goals[self.gi]) if self.gi<len(self.goals) else None
        rc=tuple(self.robot)
        for r,c in self._inf():
            if (r,c) not in (rc,gc): g[r,c]=1
        return g

    def _replan(self):
        if self.gi>=len(self.goals): self.plan=[]; self.done=True; return
        p,_=plan(self._pg(),tuple(self.robot),tuple(self.goals[self.gi]),ALGORITHM)
        self.plan=list(p[1:]) if len(p)>1 else []

    def _mobs(self):
        dirs=[(-1,0),(1,0),(0,-1),(0,1)]
        occ={(o[0],o[1]) for o in self.obs}
        rc=tuple(self.robot)
        for obs in self.obs:
            random.shuffle(dirs)
            for dr,dc in dirs:
                nr,nc=obs[0]+dr,obs[1]+dc
                if(1<=nr<GRID_ROWS-1 and 1<=nc<GRID_COLS-1
                   and self.base[nr,nc]==0
                   and (nr,nc) not in occ and (nr,nc)!=rc):
                    occ.discard((obs[0],obs[1]))
                    obs[0],obs[1]=nr,nc; occ.add((nr,nc)); break

    def step(self):
        if self.done: return
        self.fn+=1
        if self.fn%OBS_EVERY==0:
            self._mobs()
            if self.plan and any(c in self._inf() for c in self.plan[:3]):
                self._replan()
        if not self.plan: self._replan()
        if not self.plan: return
        nxt=self.plan.pop(0); self.robot=list(nxt); self.vis.append(nxt)
        if self.gi<len(self.goals):
            gr,gc=self.goals[self.gi]
            if abs(self.robot[0]-gr)+abs(self.robot[1]-gc)<=1:
                self.col.add(self.gi); self.gi+=1; self.plan=[]; self._replan()

# ─── figure ───────────────────────────────────────────────────────────────
env=Env()
fig=plt.figure(figsize=(11,7),facecolor=C_BG)
ax=fig.add_subplot(111)
ax.set_facecolor(C_BG); ax.set_xlim(0,GRID_COLS); ax.set_ylim(0,GRID_ROWS)
ax.set_aspect('equal'); ax.axis('off')

fig.suptitle(f'Grid Navigation  ·  {ALGORITHM.upper()}  '
             f'·  Goals:{len(env.goals)}  Obs:{NUM_OBS}',
             color='white',fontsize=12,fontweight='bold',y=0.97)

def ctr(r,c): return c+0.5,(GRID_ROWS-1-r)+0.5
def org(r,c): return c,GRID_ROWS-1-r

# walls
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if env.base[r,c]==1:
            ax.add_patch(mpatches.Rectangle(org(r,c),1,1,
                facecolor=C_WALL,edgecolor='#0a0a18',lw=0.2,zorder=1))
# grid lines
for i in range(GRID_COLS+1): ax.axvline(i,color='#1a2333',lw=0.25,zorder=0)
for j in range(GRID_ROWS+1): ax.axhline(j,color='#1a2333',lw=0.25,zorder=0)

plan_p=[]; vis_p=[]

obs_p=[mpatches.FancyBboxPatch((0,0),0.78,0.78,boxstyle='round,pad=0.05',
        facecolor=C_OBS,edgecolor='white',lw=0.4,alpha=0.9,zorder=5)
       for _ in env.obs]
for p in obs_p: ax.add_patch(p)

rob=plt.Circle(ctr(*ROBOT_START),0.38,facecolor=C_ROB,edgecolor='white',lw=2,zorder=8)
ax.add_patch(rob)

gstars=[]
for i,(gr,gc) in enumerate(env.goals):
    cx,cy=ctr(gr,gc)
    s,=ax.plot(cx,cy,'*',color=C_GOAL,ms=18,zorder=7,
               markeredgecolor='white',markeredgewidth=0.4)
    ax.text(cx,cy-0.52,str(i+1),color='w',fontsize=6,ha='center',
            fontweight='bold',zorder=9)
    gstars.append(s)

info=ax.text(0.01,0.98,'',transform=ax.transAxes,color='white',
             fontsize=7.5,va='top',ha='left',
             bbox=dict(boxstyle='round',facecolor='#1a1a3a',
                       edgecolor='#5566aa',alpha=0.88))

def update(frame):
    global plan_p, vis_p
    env.step()

    # orange planned
    for p in plan_p: p.remove()
    plan_p=[]
    for cell in env.plan:
        r,c=cell
        x,y=org(r,c)
        p=mpatches.Rectangle((x+0.1,y+0.1),0.8,0.8,
          facecolor=C_PLAN,alpha=0.55,edgecolor='none',zorder=2)
        ax.add_patch(p); plan_p.append(p)

    # green visited
    while len(vis_p)<len(env.vis):
        cell=env.vis[len(vis_p)]; r,c=cell; x,y=org(r,c)
        p=mpatches.Rectangle((x+0.05,y+0.05),0.9,0.9,
          facecolor=C_VIS,alpha=0.4,edgecolor='none',zorder=2)
        ax.add_patch(p); vis_p.append(p)

    # robot
    rx,ry=ctr(*env.robot); rob.set_center((rx,ry))

    # obstacles
    for i,o in enumerate(env.obs):
        ox,oy=org(o[0],o[1])
        obs_p[i].set_x(ox+0.11)
        obs_p[i].set_y(oy+0.11)

    # goals
    for i,s in enumerate(gstars):
        s.set_color(C_DONE if i in env.col else C_GOAL)
        s.set_markersize(13 if i in env.col else 18)

    nxt=(f"→ Goal {env.gi+1}: {env.goals[env.gi]}"
         if env.gi<len(env.goals) else "ALL DONE!")
    info.set_text(f"Frame:{frame+1}/{N_FRAMES}  "
                  f"Collected:{len(env.col)}/{len(env.goals)}  {nxt}")

# ─── render ───────────────────────────────────────────────────────────────
print(f"\n  Rendering {N_FRAMES} frames → {OUT}")
print("  Please wait …\n")
anim=animation.FuncAnimation(fig,update,frames=N_FRAMES,
                             interval=1000//FPS,repeat=False)
try:
    import imageio_ffmpeg
    mp4=OUT.replace('.gif','.mp4')
    plt.rcParams['animation.ffmpeg_path']=imageio_ffmpeg.get_ffmpeg_exe()
    w=animation.FFMpegWriter(fps=FPS,bitrate=1800,
                             extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])
    anim.save(mp4,writer=w,dpi=110)
    print(f"  ✓ MP4 saved: {mp4}")
    os.startfile(mp4)
except Exception as e:
    print(f"  MP4 failed ({e}) — saving GIF …")
    anim.save(OUT,writer=animation.PillowWriter(fps=FPS),dpi=100)
    print(f"  ✓ GIF saved: {OUT}")
    os.startfile(OUT)
