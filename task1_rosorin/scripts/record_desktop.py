#!/usr/bin/env python3
"""Record real X11 desktop frames with OpenCV; no generated imagery or replay."""
import argparse,json,time
from pathlib import Path
import gi
import cv2
import numpy as np
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=220)
    p.add_argument('--fps',type=int,default=5)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stages',type=Path)
    p.add_argument('--stop-file',type=Path,help='end early once this file exists')
    a=p.parse_args()
    cv2.setNumThreads(1)
    window=Gdk.get_default_root_window()
    width,height=window.get_width(),window.get_height()
    writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(width,height))
    if not writer.isOpened():raise RuntimeError('OpenCV MPEG-4 encoder unavailable')
    start=time.monotonic();timings=[];events=[];seen=0
    try:
     for i in range(round(a.seconds*a.fps)):
      if a.stop_file and a.stop_file.exists():break
      time.sleep(max(0,start+i/a.fps-time.monotonic()))
      pix=Gdk.pixbuf_get_from_window(window,0,0,width,height)
      if pix is None:raise RuntimeError('X11 desktop capture unavailable')
      arr=np.frombuffer(pix.get_pixels(),dtype=np.uint8).reshape(height,pix.get_rowstride())
      rgb=arr[:,:width*pix.get_n_channels()].reshape(height,width,pix.get_n_channels())[:,:,:3]
      writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
      timings.append(time.monotonic()-start)
      if a.stages and a.stages.exists():
       names=a.stages.read_text().splitlines()
       for name in names[seen:]:
        events.append({'stage':name,'video_seconds':i/a.fps,'capture_wall_seconds':timings[-1]})
        pix.savev(str(a.stages.parent/('screen_'+name+'.png')),'png',[],[])
       seen=len(names)
    finally:
     writer.release()
     a.output.with_suffix('.timing.json').write_text(json.dumps({'frames':len(timings),'fps':a.fps,'wall_seconds':time.monotonic()-start,'frame_capture_seconds':timings,'width':width,'height':height,'events':events},indent=2)+'\n')


if __name__ == "__main__":
    main()
