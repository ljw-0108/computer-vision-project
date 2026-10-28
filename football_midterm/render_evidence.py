"""Render compact ID-only labels from saved CSV; no model inference is repeated."""
import subprocess
from pathlib import Path
import cv2,pandas as pd
ROOT=Path(__file__).resolve().parent
for name in ['bytetrack','botsort']:
 folder=ROOT/'results'/name
 df=pd.read_csv(folder/'positions.csv');groups={int(k):v for k,v in df.groupby('frame')}
 cap=cv2.VideoCapture(str(ROOT/'data/sample.mp4'));fps=cap.get(5)
 width,height=int(cap.get(3)),int(cap.get(4))
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','26','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/'tracking_preview.mp4')],stdin=subprocess.PIPE)
 i=0
 while True:
  ret,im=cap.read()
  if not ret:break
  if i in groups:
   for row in groups[i].itertuples():
    tid=int(row.track_id); color=(int(80+(tid*47)%170),int(80+(tid*79)%170),int(80+(tid*113)%170))
    x1,y1,x2,y2=map(int,[row.x1,row.y1,row.x2,row.y2])
    cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
    label=f'ID {tid}'
    (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.52,1)
    cv2.rectangle(im,(x1,max(0,y1-th-8)),(x1+tw+5,y1),color,-1)
    cv2.putText(im,label,(x1+2,max(th,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1,cv2.LINE_AA)
    cv2.circle(im,(int(row.foot_x),int(row.foot_y)),3,color,-1)
  cv2.rectangle(im,(10,10),(740,52),(20,20,20),-1)
  cv2.putText(im,f'{name} | t={i/fps:.2f}s | person tracks, image coordinates',(20,38),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1,cv2.LINE_AA)
  if i in [0,149,374,749]:cv2.imwrite(str(folder/f'frame_{i:04d}.jpg'),im)
  proc.stdin.write(im.tobytes());i+=1
 proc.stdin.close();code=proc.wait();cap.release()
 if code:raise RuntimeError('ffmpeg failed')
 print(name,'rendered',i,flush=True)
