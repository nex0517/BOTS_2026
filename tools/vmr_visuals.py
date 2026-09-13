"""Orthogonal CT projections with parent, traced predictions and rejected openings."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image,ImageDraw,ImageFont


def visual_check(ct,mask,record,reference,destination,case_metrics):
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    except OSError:font=ImageFont.load_default();small=font
    coords=np.argwhere(mask.data>0);pad=np.ceil(15/ct.spacing).astype(int);lo=np.maximum(0,coords.min(axis=0)-pad);hi=np.minimum(ct.data.shape,coords.max(axis=0)+pad+1)
    sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));data=ct.data[sl];parent=mask.data[sl]>0
    predictions=record['prediction']['daughters'];diagnostics=record['diagnostics'];branches=diagnostics['branches'];rejected=diagnostics.get('rejected_candidates',[])
    width=1500;panel_w=480;panel_h=650;header=100
    canvas=Image.new('RGB',(width,header+panel_h+140+28*len(predictions)),(15,22,30));draw=ImageDraw.Draw(canvas)
    draw.text((18,14),f"{record['prediction']['case_id']} | Derived VMR validation | {len(predictions)} predictions / {len(reference['daughters'])} references",font=font,fill='white')
    draw.text((18,43),f"6 mm matching: TP {case_metrics['tp']}  FP {case_metrics['fp']}  FN {case_metrics['fn']}   F1 {case_metrics['f1']:.3f}",font=font,fill=(220,230,240))
    draw.text((18,72),'Green: accepted trace/origin   Cyan: seed   Red crosses: rejected openings   White squares: derived reference origins',font=small,fill=(210,220,230))
    for column,(axis,title) in enumerate([(2,'Axial: native x / y'),(1,'Coronal: native x / z'),(0,'Sagittal: native y / z')]):
        projection=np.max(data,axis=axis).T
        gray=np.uint8(np.clip((projection+100)/700,0,1)*255)
        pixels=np.repeat(gray[...,None],3,axis=2)
        footprint=parent.any(axis=axis).T;outline=footprint&~ndi.binary_erosion(footprint)
        pixels[footprint]=(pixels[footprint]*.65+np.array([25,75,125])*.35).astype(np.uint8);pixels[outline]=[75,155,235]
        remaining=[k for k in range(3) if k!=axis]
        physical_size=np.array([pixels.shape[1]*ct.spacing[remaining[0]],pixels.shape[0]*ct.spacing[remaining[1]]])
        scale=min((panel_w-20)/physical_size[0],(panel_h-35)/physical_size[1]);size=np.maximum(1,np.rint(physical_size*scale).astype(int))
        image=Image.fromarray(pixels).resize(tuple(size),Image.Resampling.BILINEAR)
        left=10+column*500+(panel_w-size[0])//2;top=header+30
        canvas.paste(image,(int(left),int(top)))
        draw.text((column*500+18,header),title,font=font,fill='white')
        def point(world):
            index=ct.physical_to_index(world)[0]-lo
            return tuple(float(v) for v in np.array([left,top])+index[remaining]*ct.spacing[remaining]*scale)
        for candidate in rejected:
            if 'ostium_xyz_mm' not in candidate:continue
            x,y=point(candidate['ostium_xyz_mm']);draw.line((x-3,y-3,x+3,y+3),fill=(245,90,90),width=1);draw.line((x-3,y+3,x+3,y-3),fill=(245,90,90),width=1)
        for r in reference['daughters']:
            x,y=point(r['ostium_xyz_mm']);draw.rectangle((x-5,y-5,x+5,y+5),outline='white',width=1)
        for d,b in zip(predictions,branches):
            path=[point(p) for p in b.get('path_xyz_mm',[d['ostium_xyz_mm'],d['seed_xyz_mm']])]
            draw.line(path,fill=(75,240,140),width=2)
            x,y=point(d['ostium_xyz_mm']);draw.ellipse((x-4,y-4,x+4,y+4),outline=(75,240,140),width=2)
            sx,sy=point(d['seed_xyz_mm']);draw.ellipse((sx-3,sy-3,sx+3,sy+3),fill=(55,225,255))
            ex,ey=point(np.array(d['seed_xyz_mm'])+2*np.array(d['direction_xyz']))
            draw.line((sx,sy,ex,ey),fill=(55,225,255),width=2)
            delta=np.array([ex-sx,ey-sy]);norm=np.linalg.norm(delta)
            if norm>0:
                u=delta/norm;v=np.array([-u[1],u[0]]);tip=np.array([ex,ey]);draw.polygon([tuple(tip),tuple(tip-7*u+3*v),tuple(tip-7*u-3*v)],fill=(55,225,255))
    y=header+panel_h+12
    draw.text((18,y),'Predicted origins in SimpleITK physical LPS mm (rounded here; JSON retains full precision)',font=font,fill='white');y+=30
    matched={m['prediction'] for m in case_metrics['matches']}
    for d in predictions:
        x0,y0,z0=d['ostium_xyz_mm'];status='matched' if d['instance_id'] in matched else 'unmatched'
        draw.text((18,y),f"{d['instance_id']}   ({x0:.3f}, {y0:.3f}, {z0:.3f})    radius {d['radius_mm']:.3f} mm    {status}",font=small,fill=(75,240,140) if status=='matched' else (255,180,95));y+=28
    draw.text((18,y+12),'Maximum-intensity projections aid overview; overlap in a projection does not prove a 3D connection. VMR references are derived, not organizer ground truth.',font=small,fill=(185,198,210))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);canvas.save(destination)
