"""Filter a tracking CSV by ID and half-open time interval [start, end)."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser()
p.add_argument('--csv',default='results/bytetrack/positions.csv')
p.add_argument('--id',type=int,required=True)
p.add_argument('--start',type=float,default=0)
p.add_argument('--end',type=float,default=30)
p.add_argument('--metadata',default='results/experiment.json')
p.add_argument('--output',default='results/selected')
a=p.parse_args()
if a.end<=a.start: p.error('end must be greater than start')
m=json.loads(Path(a.metadata).read_text())['video'];fps=m['fps'];w=m['width'];h=m['height']
d=pd.read_csv(a.csv)
g=d[(d.track_id==a.id)&(d.time_s>=a.start)&(d.time_s<a.end)].sort_values('frame')
if g.empty: raise SystemExit('No observations for this ID/time interval')
o=Path(a.output);o.mkdir(parents=True,exist_ok=True);g.to_csv(o/'selected_positions.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(11,4))
for _,s in g.groupby((g.frame.diff().fillna(1)>1).cumsum()): ax[0].plot(s.foot_x,s.foot_y,lw=1)
ax[0].scatter(g.foot_x,g.foot_y,s=4,c=g.time_s,cmap='viridis')
z,_,_=np.histogram2d(g.foot_x,g.foot_y,bins=[48,27],range=[[0,w],[0,h]])
im=ax[1].imshow(z.T/fps,extent=[0,w,h,0],aspect='auto',cmap='magma');fig.colorbar(im,ax=ax[1],label='Observed seconds / bin')
for q in ax:q.set(xlim=(0,w),ylim=(h,0),xlabel='Image x (px)',ylabel='Image y (px)')
ax[0].set_title('Path (gaps not connected)');ax[1].set_title('Screen occupancy')
fig.suptitle(f'Track ID {a.id}, [{a.start}, {a.end}) s - screen coordinates, not pitch')
fig.tight_layout();fig.savefig(o/'selected_analysis.png',dpi=160)
print(json.dumps({'track_id':a.id,'rows':len(g),'observed_seconds':len(g)/fps,'first_observed_time':float(g.time_s.min()),'last_observed_time':float(g.time_s.max())},indent=2))
