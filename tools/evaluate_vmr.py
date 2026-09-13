"""Compare image-only detector versions with read-only derived VMR references."""
import argparse,json,os,sys,time,resource,subprocess
from pathlib import Path


def worker(args):
    affinity=False
    if hasattr(os,'sched_setaffinity'):
        cpus=sorted(os.sched_getaffinity(0))[:4]
        os.sched_setaffinity(0,cpus);affinity=len(cpus)==4
    if args.require_affinity and not affinity:
        raise RuntimeError('Four-core CPU affinity is unavailable on this platform')
    sys.path.insert(0,str(Path(args.implementation).resolve()))
    from detector.branchseed import read_nifti,detect,validate_output
    from detector.tracking import validate_trace
    start=time.perf_counter()
    ct=read_nifti(Path(args.data)/f'patient_{args.worker}_image.nii')
    mask=read_nifti(Path(args.data)/f'patient_{args.worker}_binary_mask.nii')
    result,diag=detect(ct,mask);elapsed=time.perf_counter()-start
    result['case_id']=f'patient_{args.worker}';validate_output(result)
    for branch in diag['branches']:
        if branch.get('path_status')=='traced':validate_trace(branch)
        elif args.strict_traces:raise ValueError('Output contains an unresolved branch')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    record=dict(prediction=result,diagnostics=diag,runtime_s=elapsed,peak_rss_mib=rss,four_core_affinity=affinity)
    (folder/f'patient_{args.worker}.json').write_text(json.dumps(record,indent=2)+'\n')


def metrics(records,refs,tolerance):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    rows=[];all_matches=[]
    def summarize(tp,fp,fn):
        p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
        return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    def geometry(matches):
        keys=['ostium_error_mm','ostium_abs_xyz_mm','seed_error_mm','seed_abs_xyz_mm','radius_error_mm','direction_error_degrees','direction_abs_xyz']
        return {k:(np.mean([m[k] for m in matches if m[k] is not None],axis=0).tolist() if any(m[k] is not None for m in matches) else None) for k in keys}
    for index,(record,ref) in enumerate(zip(records,refs),1):
        ds=record['prediction']['daughters'];rs=ref['daughters']
        distances=np.array([[np.linalg.norm(np.array(d['ostium_xyz_mm'])-r['ostium_xyz_mm']) for r in rs] for d in ds]).reshape(len(ds),len(rs))
        penalty=(min(distances.shape)+1)*(tolerance+1)
        ii,jj=linear_sum_assignment(np.where(distances<=tolerance,distances,penalty))
        pairs=[(i,j) for i,j in zip(ii,jj) if distances[i,j]<=tolerance];matches=[]
        for i,j in pairs:
            d,r=ds[i],rs[j];oe=np.abs(np.array(d['ostium_xyz_mm'])-r['ostium_xyz_mm']);se=np.abs(np.array(d['seed_xyz_mm'])-r['seed_xyz_mm'])
            a=np.array(d['direction_xyz']);b=np.array(r['direction_xyz']);dot=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            matches.append(dict(prediction=d['instance_id'],reference=r['instance_id'],ostium_error_mm=float(np.linalg.norm(oe)),ostium_abs_xyz_mm=oe.tolist(),seed_error_mm=float(np.linalg.norm(se)),seed_abs_xyz_mm=se.tolist(),radius_error_mm=abs(d['radius_mm']-r['radius_mm']) if r.get('radius_mm') is not None else None,direction_error_degrees=float(np.degrees(np.arccos(np.clip(dot,-1,1)))),direction_abs_xyz=np.abs(a-b).tolist(),direction_signed_xyz=(a-b).tolist(),ostium_signed_xyz_mm=(np.array(d['ostium_xyz_mm'])-r['ostium_xyz_mm']).tolist(),seed_signed_xyz_mm=(np.array(d['seed_xyz_mm'])-r['seed_xyz_mm']).tolist()))
        all_matches+=matches
        rows.append(dict(case=index,references=len(rs),predictions=len(ds),**summarize(len(pairs),len(ds)-len(pairs),len(rs)-len(pairs)),geometry=geometry(matches),matches=matches,unmatched_predictions=[d['instance_id'] for i,d in enumerate(ds) if i not in {i for i,j in pairs}],missed_references=[r['instance_id'] for j,r in enumerate(rs) if j not in {j for i,j in pairs}]))
    overall=summarize(*[sum(r[k] for r in rows) for k in ['tp','fp','fn']])
    overall.update(references=sum(len(r['daughters']) for r in refs),predictions=sum(len(r['prediction']['daughters']) for r in records),geometry=geometry(all_matches))
    return dict(tolerance_mm=tolerance,overall=overall,cases=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--implementation',required=True);parser.add_argument('--data',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--strict-traces',action='store_true');parser.add_argument('--worker',type=int);parser.add_argument('--reuse',action='store_true');parser.add_argument('--require-affinity',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args)
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']:env[key]='4'
    for n in range(1,6):
        if not args.reuse:
            cmd=[sys.executable,__file__,'--implementation',args.implementation,'--data',args.data,'--output',args.output,'--worker',str(n)]
            if args.require_affinity:cmd.append('--require-affinity')
            if args.strict_traces:cmd.append('--strict-traces')
            subprocess.run(cmd,env=env,check=True)
    records=[json.loads((Path(args.output)/f'patient_{n}.json').read_text()) for n in range(1,6)]
    refs=[json.loads((Path(args.data)/f'patient_{n}_test.json').read_text()) for n in range(1,6)]
    report=dict(methodology='Derived VMR geometry, not organizer ground truth. Maximum-cardinality/minimum-distance one-to-one matching. Numerical libraries limited to four threads.',performance=[{k:r[k] for k in ['runtime_s','peak_rss_mib','four_core_affinity']} for r in records],tolerances=[metrics(records,refs,t) for t in [3.,6.,10.]])
    (Path(args.output)/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for result in report['tolerances']:
        print(result['tolerance_mm'],json.dumps(result['overall']))
        if result['tolerance_mm']==6:print([(r['case'],r['predictions'],r['tp'],r['fp'],r['fn']) for r in result['cases']])
    print('mean_runtime',sum(r['runtime_s'] for r in records)/5,'peak_rss',max(r['peak_rss_mib'] for r in records),flush=True)
if __name__=='__main__':main()
