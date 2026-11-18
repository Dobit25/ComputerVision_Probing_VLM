"""Outcome-blind review and a new source-disjoint 2x2 confirmation set."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import math
from statistics import NormalDist

import numpy as np

from ..io_utils import sha256, write_json, source_snapshot, configure_paths, command_output
from .common import ROOT, read_json, write_csv, finish, cluster_ci
from . import natural
from .design import spatial_trials
from .pairs import assignment_gap

CONFIG = ROOT/'configs/natural_confirmation_v1.json'
PROTOCOL = ROOT/'docs/natural_confirmation_protocol.md'
SELECTION = ROOT/'results/natural_confirmation_selection_v1'
SHEETS = ROOT/'data/natural_confirmation_v1/review_sheets'


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen artifact differs: {path}')
    else:
        write_json(path, value)


def sheets(rows, prefix):
    """Reuse rendering, with separate files and no previous review decisions shown."""
    SHEETS.mkdir(parents=True, exist_ok=True)
    original_data = natural.DATA
    try:
        natural.DATA = SHEETS.parent
        natural.contact_sheets(rows, prefix)
    finally:
        natural.DATA = original_data


def dhash(row, mirror=False):
    from PIL import Image, ImageOps
    with Image.open(ROOT/row['image_path']) as im:
        im = im.convert('RGB')
        side = min(im.size); x,y = (im.width-side)/2,(im.height-side)/2
        im = im.crop((x,y,x+side,y+side))
        if mirror:
            im = ImageOps.mirror(im)
        v = np.asarray(im.convert('L').resize((9,8)))
    return v[:,1:] > v[:,:-1]


def prepare_old():
    SELECTION.mkdir(parents=True, exist_ok=True)
    cfg = read_json(CONFIG)
    rows = read_json(ROOT/cfg['old_selected'])
    rows = [{**r, 'original_review_index':r['review_index'], 'review_index':i} for i,r in enumerate(rows)]
    frozen_json(SELECTION/'old_candidates.json', rows)
    sheets(rows, 'old')
    print('Prepared 200 old images for second review, without per-image scores.', flush=True)


def prepare_new():
    SELECTION.mkdir(parents=True, exist_ok=True)
    cfg = read_json(CONFIG); base = read_json(ROOT/cfg['base_config'])
    excluded = read_json(ROOT/cfg['exclude_all_previous_candidates'])
    if (SELECTION/'new_candidates.json').exists():
        rows = read_json(SELECTION/'new_candidates.json')
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(natural.download_image, rows))
        sheets(rows, 'new')
        print('Restored fixed new candidates.', flush=True)
        return
    annotation = natural.annotations(base)
    excluded_ids = {r['source_image_id'] for r in excluded}
    annotation['images'] = [r for r in annotation['images'] if r['id'] not in excluded_ids]
    nc = {**base['natural'], 'allowed_relations':cfg['allowed_relations'],
          'selection_seed':cfg['selection_seed'], 'candidate_reserve_per_relation':100000}
    rows, counts = natural.candidates(annotation,nc)
    frozen_json(SELECTION/'candidate_metadata.json',rows)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(natural.download_image,rows))
    for i,r in enumerate(rows):
        r['review_index'] = i
    frozen_json(SELECTION/'new_candidates.json',rows)
    old_hashes = np.stack([dhash(r) for r in excluded])
    old_flips = np.stack([dhash(r,True) for r in excluded])
    previous, duplicates = [], []
    for r in rows:
        h = dhash(r)
        distances = np.minimum(np.count_nonzero(old_hashes!=h,axis=(1,2)), np.count_nonzero(old_flips!=h,axis=(1,2)))
        best = int(distances.argmin())
        if distances[best] <= cfg['near_duplicate_dhash_max_distance']:
            duplicates.append({'review_index':r['review_index'],'source_image_id':r['source_image_id'],
                'matches_previous_source_id':excluded[best]['source_image_id'],'distance':int(distances[best])})
            continue
        match = next(((v, int(min(np.count_nonzero(h!=a),np.count_nonzero(h!=b)))) for v,a,b in previous
                      if min(np.count_nonzero(h!=a),np.count_nonzero(h!=b)) <= cfg['near_duplicate_dhash_max_distance']),None)
        if match:
            duplicates.append({'review_index':r['review_index'],'source_image_id':r['source_image_id'],
                               'matches_new_source_id':match[0],'distance':match[1]})
        else:
            previous.append((r['source_image_id'],h,dhash(r,True)))
    frozen_json(SELECTION/'duplicate_screen.json',duplicates)
    frozen_json(SELECTION/'preparation.json',{'prepared_at_utc':now(),'counts':counts,
        'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL),
        'excluded_candidates_sha256':sha256(ROOT/cfg['exclude_all_previous_candidates']),
        'annotation_source':read_json(natural.SELECTION/'source.json'),
        'new_candidate_sha256':sha256(SELECTION/'new_candidates.json')})
    sheets(rows,'new')
    print({'new_candidates':len(rows),'duplicate_flags':duplicates,'counts':counts},flush=True)


def freeze_review(kind, exclusions_path):
    """Exclusions file is explicitly authored after actually viewing every sheet."""
    if kind not in ('old','new'):
        raise ValueError('Invalid review kind.')
    if (SELECTION/f'{kind}_review.json').exists():
        raise FileExistsError('Review already frozen.')
    candidates = read_json(SELECTION/f'{kind}_candidates.json')
    exclusions = read_json(exclusions_path)
    if any(int(k) not in range(len(candidates)) or not str(v).strip() for k,v in exclusions.items()):
        raise ValueError('Invalid review exclusion index/reason.')
    if kind == 'new':
        for r in read_json(SELECTION/'duplicate_screen.json'):
            exclusions.setdefault(str(r['review_index']), 'Near-duplicate flagged before inference: '+str(r))
    decisions = [{ 'review_index':r['review_index'],'source_image_id':r['source_image_id'],
        'image_sha256':r['image_sha256'],'accepted':str(r['review_index']) not in exclusions,
        'reason':exclusions.get(str(r['review_index']), 'Both targets, class names and relation confirmed in the center crop.')} for r in candidates]
    review = {'reviewed_at_utc':now(),'reviewer':'Codex assistant; same reviewer, not independent human annotation',
              'kind':kind,'per_image_clip_scores_shown':False,'old_aggregate_results_known':True,
              'candidate_sha256':sha256(SELECTION/f'{kind}_candidates.json'),
              'sheet_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(SHEETS.glob(f'{kind}_*.png'))},
              'decisions':decisions}
    write_json(SELECTION/f'{kind}_review.json',review)
    print({'kind':kind,'reviewed':len(decisions),'accepted':sum(r['accepted'] for r in decisions)},flush=True)


def selected(kind):
    rows = read_json(SELECTION/f'{kind}_candidates.json')
    review = read_json(SELECTION/f'{kind}_review.json')
    if review['candidate_sha256'] != sha256(SELECTION/f'{kind}_candidates.json'):
        raise ValueError('Reviewed candidate bytes changed.')
    if len(review['decisions']) != len(rows):
        raise ValueError('Incomplete review.')
    kept=[]
    for r,d in zip(rows,review['decisions']):
        if (r['source_image_id'],r['review_index'],r['image_sha256']) != (d['source_image_id'],d['review_index'],d['image_sha256']):
            raise ValueError('Review order or image mismatch.')
        if d['accepted']:
            kept.append(r)
    if kind=='new':
        cfg=read_json(CONFIG)
        if len(kept)<cfg['minimum_new_sources']:
            raise ValueError('Too few accepted sources; stop before inference.')
        kept=kept[:cfg['target_new_sources']]
    return kept


def freeze_selection():
    rows=selected('new')
    frozen_json(SELECTION/'selected.json',rows)
    if not (SELECTION/'freeze.json').exists():
        write_json(SELECTION/'freeze.json',{'frozen_at_utc':now(),'num_sources':len(rows),
            'sha256':{p.name:sha256(p) for p in sorted(SELECTION.iterdir()) if p.is_file()},
            'config_sha256':sha256(CONFIG),'protocol_sha256':sha256(PROTOCOL)})
    print('New selection frozen:',len(rows),flush=True)


def wilson(k,n):
    if n<=0 or k<0 or k>n:
        raise ValueError('Invalid binomial counts.')
    z=NormalDist().inv_cdf(.975); p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [100*(center-half),100*(center+half)]


def paired_metrics(trials,scores,config,keep_ids=None):
    groups={}
    for i,r in enumerate(trials):
        if r['target'] not in (0,1) or (keep_ids is not None and r['group_id'] not in keep_ids):
            continue
        key=(r['group_id'],r['template'],r['order'])
        groups.setdefault(key,[]).append((i,r))
    records=[]
    for (group,t,o),g in groups.items():
        g.sort(key=lambda x:x[1]['target'])
        if len(g)!=2 or [r['target'] for _,r in g] != [0,1]:
            raise ValueError('Invalid horizontal source pair.')
        m=np.asarray(scores[[i for i,_ in g],:2],dtype=np.float64)
        gap=assignment_gap(m); decisions=m.argmax(1)
        records.append({'group_id':group,'template':t,'order':o,'gap':gap,'correct':gap>0,
            'exact_tie':gap==0,'near_tie':abs(gap)<=1e-6,'two_way_correct_count':int((decisions==[0,1]).sum()),
            'both_images_correct':bool(np.array_equal(decisions,[0,1]))})
    summary={}
    for t in range(3):
        for o in range(2):
            r=[x for x in records if x['template']==t and x['order']==o]; n=len(r)
            if not n: raise ValueError('Empty condition.')
            k=sum(x['correct'] for x in r)
            summary[f't{t}_o{o}']={'n_sources':n,'correct':k,'accuracy_percent':100*k/n,
                'wilson_ci95_percent':wilson(k,n),
                'bootstrap_ci95_percent':cluster_ci([x['correct'] for x in r],[x['group_id'] for x in r],config['bootstrap_draws'],config['bootstrap_seed']),
                'binomial_p_greater_0_5':sum(math.comb(n,j) for j in range(k,n+1))/2**n,
                'two_way_accuracy_percent':50*sum(x['two_way_correct_count'] for x in r)/n,
                'both_images_correct_percent':100*sum(x['both_images_correct'] for x in r)/n,
                'mean_gap':float(np.mean([x['gap'] for x in r])),
                'exact_ties':sum(x['exact_tie'] for x in r),'near_ties':sum(x['near_tie'] for x in r)}
    return summary,records


def validate_inputs(rows):
    cfg=read_json(CONFIG); base=read_json(ROOT/cfg['base_config'])
    excluded={r['source_image_id'] for r in read_json(ROOT/cfg['exclude_all_previous_candidates'])}
    if len({r['source_image_id'] for r in rows})!=len(rows) or excluded & {r['source_image_id'] for r in rows}:
        raise ValueError('Repeated or previously screened source.')
    ann=natural.annotations(base); anns={r['id']:r for r in ann['annotations']}
    names={r['id']:r['name'] for r in ann['categories']}
    counts=Counter((r['image_id'],r['category_id']) for r in ann['annotations'])
    for r in rows:
        a,b=anns[r['annotation_a']],anns[r['annotation_b']]
        if a['image_id']!=r['source_image_id'] or b['image_id']!=r['source_image_id']:
            raise ValueError('Wrong source annotation.')
        if a['bbox']!=r['bbox_a'] or b['bbox']!=r['bbox_b'] or names[a['category_id']]!=r['category_a'] or names[b['category_id']]!=r['category_b']:
            raise ValueError('Wrong boxes/category names.')
        if a['iscrowd'] or b['iscrowd'] or a['category_id']==b['category_id'] or counts[(a['image_id'],a['category_id'])]!=1 or counts[(b['image_id'],b['category_id'])]!=1:
            raise ValueError('Invalid target object multiplicity.')
        if natural.relation_from_boxes(a['bbox'],b['bbox'],r['width'],r['height'],base['natural']) != r['target'] or r['target'] not in (0,1):
            raise ValueError('Relation does not match geometry.')
        if sha256(ROOT/r['image_path'])!=r['image_sha256']:
            raise ValueError('Source image checksum differs.')


def run(run_name=None):
    cfg=read_json(CONFIG); base=read_json(ROOT/cfg['base_config'])
    rows=read_json(SELECTION/'selected.json')
    if rows != selected('new'):
        raise ValueError('Selection differs from ordered accepted candidates.')
    freeze=read_json(SELECTION/'freeze.json')
    if sha256(CONFIG)!=freeze['config_sha256'] or sha256(PROTOCOL)!=freeze['protocol_sha256']:
        raise ValueError('Protocol changed after selection freeze.')
    for name,h in freeze['sha256'].items():
        if sha256(SELECTION/name)!=h:raise ValueError('Frozen selection artifact changed.')
    validate_inputs(rows)
    configure_paths(ROOT)
    import re
    name=run_name or 'full_natural_confirmation_v1'
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}',name):raise ValueError('Invalid run name.')
    out=ROOT/'results'/name
    out.mkdir(exist_ok=False)
    hashes,snapshot=source_snapshot(ROOT,[CONFIG,PROTOCOL,ROOT/cfg['base_config'],ROOT/cfg['model_config']])
    manifest={'stage':'natural_confirmation','mode':'full','status':'running','started_at_utc':now(),
        'git_head':command_output(['git','rev-parse','HEAD'],ROOT),'source_sha256':hashes,'source_snapshot':snapshot,
        'input_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(SELECTION.iterdir()) if p.is_file()}}
    write_json(out/'config.json',cfg); write_json(out/'evaluation_config.json',base)
    (out/'protocol.md').write_bytes(PROTOCOL.read_bytes());write_json(out/'manifest.json',manifest)
    from .common import Engine,ImageFiles
    images=natural.make_images(rows); trials=spatial_trials(images,base,'photo')
    engine=Engine(out,base,manifest)
    pool=list(dict.fromkeys(c for r in trials for c in r['captions'])); indexes={c:i for i,c in enumerate(pool)}
    features=engine.images(ImageFiles(images,engine.adapter.preprocess),{'images':[{k:r.get(k) for k in ['image_id','image_sha256','transform']} for r in images]})
    scores_all=engine.scores(features,engine.texts(pool))
    scores=np.array([[scores_all[r['image_index'],indexes[c]] for c in r['captions']] for r in trials],dtype=np.float32)
    metrics,records=paired_metrics(trials,scores,cfg)
    write_json(out/'images.json',images);write_json(out/'trials.json',trials);write_json(out/'text_pool.json',pool)
    np.savez_compressed(out/'scores.npz',scores=scores)
    write_json(out/'metrics.json',metrics);write_csv(out/'pairs.csv',records)
    write_csv(out/'predictions.csv',[{'sample_id':r['sample_id'],'target':r['target'],'four_way_prediction':int(s.argmax()),'two_way_prediction':int(s[:2].argmax())} for r,s in zip(trials,scores)])
    manifest.update(num_source_images=len(rows),num_images=len(images),num_trials=len(trials))
    finish(out,manifest);print(metrics['t2_o0'],flush=True)


def audit(run_name=None):
    from .audit import equal
    import zipfile
    out=ROOT/'results'/(run_name or 'full_natural_confirmation_v1')
    m=read_json(out/'manifest.json')
    if m['status']!='completed':raise ValueError('Run incomplete.')
    if {p.name for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}!=set(m['artifact_sha256']):raise ValueError('Unexpected artifacts.')
    for name,h in m['artifact_sha256'].items():
        if sha256(out/name)!=h:raise ValueError('Changed result artifact.')
    for name,h in m['input_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('Changed reviewed input.')
    snapshot=ROOT/m['source_snapshot']['path']
    if sha256(snapshot)!=m['source_snapshot']['sha256']:raise ValueError('Changed source archive.')
    with zipfile.ZipFile(snapshot) as z:
        if set(z.namelist())!=set(m['source_sha256']):raise ValueError('Wrong source archive files.')
        for name,h in m['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Changed archived code.')
        import json
        equal(read_json(out/'config.json'),json.loads(z.read('configs/natural_confirmation_v1.json')),'archived_config')
        equal(read_json(out/'evaluation_config.json'),json.loads(z.read('configs/probing_depth_v1.json')),'archived_evaluation_config')
    if sha256(out/'protocol.md')!=m['source_sha256']['docs/natural_confirmation_protocol.md']:
        raise ValueError('Run protocol differs from archived design.')
    rows=selected('new');validate_inputs(rows)
    equal(read_json(out/'images.json'),natural.make_images(rows))
    trials=spatial_trials(natural.make_images(rows),read_json(out/'evaluation_config.json'),'photo')
    equal(trials,read_json(out/'trials.json'))
    with np.load(out/'scores.npz',allow_pickle=False) as z:scores=z['scores']
    if scores.shape!=(len(trials),4) or not np.isfinite(scores).all() or (abs(scores)>1.00001).any():raise ValueError('Invalid scores.')
    metrics,records=paired_metrics(trials,scores,read_json(out/'config.json'))
    equal(metrics,read_json(out/'metrics.json'))
    from .common import csv_rows
    if [{k:str(v) for k,v in r.items()} for r in records]!=csv_rows(out/'pairs.csv'):raise ValueError('Pair records differ.')
    expected=[{'sample_id':r['sample_id'],'target':str(r['target']),'four_way_prediction':str(int(s.argmax())),'two_way_prediction':str(int(s[:2].argmax()))} for r,s in zip(trials,scores)]
    if expected!=csv_rows(out/'predictions.csv'):raise ValueError('Predictions differ.')
    frozen=read_json(SELECTION/'freeze.json')
    if frozen['frozen_at_utc']>=m['started_at_utc']:raise ValueError('Selection not frozen before inference.')
    result={'verified':True,'metrics_recomputed':True,'sources_disjoint_from_all_400_previous_candidates':True,
            'num_sources':len(rows),'selection_frozen_before_inference':True,'primary':metrics['t2_o0']}
    write_json(out.parent/(out.name+'_audit.json'),result);print(result,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare-old','prepare-new','review-old','review-new','freeze','run','audit','report'])
    p.add_argument('--exclusions');p.add_argument('--run-name');a=p.parse_args()
    if a.action=='prepare-old':prepare_old()
    elif a.action=='prepare-new':prepare_new()
    elif a.action.startswith('review-'):freeze_review(a.action.split('-')[1],a.exclusions)
    elif a.action=='freeze':freeze_selection()
    elif a.action=='run':run(a.run_name)
    elif a.action=='audit':audit(a.run_name)
    elif a.action=='report':
        from .confirmation_report import run as report
        report()


if __name__=='__main__':main()
