"""Reproducible football tracking baseline; screen coordinates, not pitch metrics."""
import argparse, json, time, platform, hashlib, importlib.metadata
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO


def run(video, weights, out, tracker, limit, threads):
    torch.set_num_threads(threads)
    cv2.setNumThreads(1)
    folder = out / tracker
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: raise ValueError('Invalid FPS')
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = min(total, limit) if limit else total
    model = YOLO(str(weights))
    # Warm up detector only, before tracker state is created.
    model.predict(np.zeros((h,w,3),np.uint8),imgsz=640,device='cpu',verbose=False)
    writer = cv2.VideoWriter(str(folder/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened(): raise RuntimeError('Cannot create video')
    rows=[]; frame_rows=[]; processing=[]; brightness=[]; differences=[]; previous=None
    t_start=time.perf_counter()
    for idx in range(count):
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        differences.append(float(np.abs(gray.astype(float)-previous).mean()) if previous is not None else 0.)
        previous=gray.astype(float)
        t=time.perf_counter()
        result=model.track(frame,persist=True,tracker=tracker+'.yaml',classes=[0],conf=0.1,iou=0.7,imgsz=640,device='cpu',verbose=False)[0]
        processing.append(time.perf_counter()-t)
        b=result.boxes
        n=0
        if b.id is not None:
            for box,tid,conf in zip(b.xyxy.cpu().numpy(),b.id.int().cpu().numpy(),b.conf.cpu().numpy()):
                x1,y1,x2,y2=map(float,box)
                rows.append(dict(frame=idx,time_s=idx/fps,track_id=int(tid),confidence=float(conf),x1=x1,y1=y1,x2=x2,y2=y2,foot_x=(x1+x2)/2,foot_y=y2,center_x=(x1+x2)/2,center_y=(y1+y2)/2))
                n+=1
        frame_rows.append(dict(frame=idx,time_s=idx/fps,tracked_persons=n,processing_ms=processing[-1]*1000,mean_luma=brightness[-1],frame_difference=differences[-1]))
        annotated=result.plot()
        cv2.putText(annotated,f'{tracker} | frame {idx} | screen coordinates',(30,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        writer.write(annotated)
        if idx in {0,149,374,749}: cv2.imwrite(str(folder/f'frame_{idx:04d}.jpg'),annotated)
        if idx==0: cv2.imwrite(str(out/'first_frame.jpg'),frame)
        if (idx+1)%150==0: print(tracker,idx+1,'/',count,flush=True)
    elapsed=time.perf_counter()-t_start
    writer.release();cap.release()
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No tracks; cannot analyze')
    frames=pd.DataFrame(frame_rows)
    df.to_csv(folder/'positions.csv',index=False)
    frames.to_csv(folder/'frames.csv',index=False)
    summaries=[]
    for tid,g in df.groupby('track_id'):
        g=g.sort_values('frame'); delta=g.frame.diff(); gap=delta[delta>1]-1
        summaries.append(dict(track_id=int(tid),observations=len(g),first_frame=int(g.frame.min()),last_frame=int(g.frame.max()),observed_seconds=len(g)/fps,internal_gap_frames=int(gap.sum()),internal_gap_events=len(gap),mean_confidence=float(g.confidence.mean())))
    tracks=pd.DataFrame(summaries).sort_values(['observations','track_id'],ascending=[False,True])
    tracks.to_csv(folder/'track_summary.csv',index=False)
    selected=int(tracks.iloc[0].track_id)
    g=df[df.track_id==selected].sort_values('frame')
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    seg=(g.frame.diff().fillna(1)>1).cumsum()
    for _,part in g.groupby(seg): axs[0].plot(part.foot_x,part.foot_y,lw=1.3,color='#226cba')
    sc=axs[0].scatter(g.foot_x,g.foot_y,c=g.time_s,s=4,cmap='viridis')
    fig.colorbar(sc,ax=axs[0],label='Time (s)')
    axs[0].set_title(f'Track {selected}: foot-point path')
    hist,xedges,yedges=np.histogram2d(g.foot_x,g.foot_y,bins=[48,27],range=[[0,w],[0,h]])
    im=axs[1].imshow(hist.T/fps,extent=[0,w,h,0],cmap='magma',aspect='auto')
    fig.colorbar(im,ax=axs[1],label='Observed time (s / bin)')
    axs[1].set_title(f'Track {selected}: screen occupancy')
    for ax in axs: ax.set(xlim=(0,w),ylim=(h,0),xlabel='Image x (pixels)',ylabel='Image y (pixels)')
    fig.suptitle('Screen coordinates only - camera motion not corrected',fontsize=11)
    fig.tight_layout();fig.savefig(folder/'path_heatmap.png',dpi=160);plt.close(fig)
    heights=df.y2-df.y1
    metrics=dict(tracker=tracker,frames=len(frames),rows=len(df),unique_ids=len(tracks),mean_tracked_persons=float(frames.tracked_persons.mean()),min_tracked_persons=int(frames.tracked_persons.min()),max_tracked_persons=int(frames.tracked_persons.max()),zero_track_frames=int((frames.tracked_persons==0).sum()),median_track_observations=float(tracks.observations.median()),short_ids_lt_1s=int((tracks.observations<fps).sum()),internal_gap_frames=int(tracks.internal_gap_frames.sum()),selected_track_id=selected,selected_track_observations=len(g),mean_confidence=float(df.confidence.mean()),bbox_height_p10=float(heights.quantile(.1)),bbox_height_median=float(heights.median()),processing_seconds=sum(processing),processing_fps=len(frames)/sum(processing),end_to_end_seconds=elapsed,end_to_end_fps=len(frames)/elapsed)
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2))
    # Evidence of tracker settings actually used, including ReID flag.
    import ultralytics
    config=Path(ultralytics.__file__).parent/'cfg'/'trackers'/(tracker+'.yaml')
    (folder/'tracker_config.yaml').write_text(config.read_text())
    return metrics,dict(width=w,height=h,fps=fps,total_frames=total,duration_seconds=total/fps,size_bytes=video.stat().st_size,sha256=hashlib.sha256(video.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',default='data/sample.mp4');p.add_argument('--weights',default='yolo11s.pt');p.add_argument('--output',default='results');p.add_argument('--trackers',nargs='+',default=['bytetrack','botsort']);p.add_argument('--limit',type=int,default=0);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for tracker in a.trackers:
        metric,video=run(Path(a.video),Path(a.weights),out,tracker,a.limit,a.threads);results.append(metric)
    pd.DataFrame(results).to_csv(out/'comparison.csv',index=False)
    env={'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(),'torch_threads':a.threads,'cuda_available':torch.cuda.is_available(),'versions':{m:importlib.metadata.version(m) for m in ['ultralytics','torch','torchvision','opencv-python','pandas','numpy','matplotlib','lap']}}
    (out/'experiment.json').write_text(json.dumps({'video':video,'environment':env,'settings':{'model':'YOLO11s COCO pretrained','classes':[0],'conf':.1,'iou':.7,'imgsz':640,'device':'cpu','stride':1,'warmup':'one black frame predict before timer','note':'Sequential single runs, no accuracy ground truth; ReID disabled by default'},'results':results},indent=2))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for tracker in a.trackers:
        d=pd.read_csv(out/tracker/'frames.csv');tr=pd.read_csv(out/tracker/'track_summary.csv')
        axs[0,0].plot(d.time_s,d.tracked_persons,label=tracker,lw=1)
        axs[0,1].hist(tr.observations/video['fps'],bins=np.linspace(0,video['duration_seconds'],16),alpha=.5,label=tracker)
    axs[0,0].set(xlabel='Time (s)',ylabel='Tracked person boxes / frame',title='Tracking output count (not accuracy)');axs[0,0].legend()
    axs[0,1].set(xlabel='Observed duration per ID (s)',ylabel='Number of IDs',title='Track duration distribution');axs[0,1].legend()
    d=pd.read_csv(out/a.trackers[0]/'frames.csv')
    axs[1,0].plot(d.time_s,d.mean_luma,color='#267a65');axs[1,0].set(xlabel='Time (s)',ylabel='Mean grayscale intensity',title='Video brightness')
    axs[1,1].plot(d.time_s,d.frame_difference,color='#a85529');axs[1,1].set(xlabel='Time (s)',ylabel='Mean absolute pixel difference',title='Frame change (not a shot-change classifier)')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=160);plt.close(fig)
    print(json.dumps({'video':video,'results':results},indent=2),flush=True)
if __name__=='__main__':main()
