"""Report the source-disjoint confirmation and post-hoc label sensitivity separately."""
import numpy as np
from ..io_utils import sha256,write_json
from .common import ROOT,read_json,write_csv
from .confirmation import CONFIG,SELECTION,selected,paired_metrics
from .spatial import compute_metrics


def run():
    cfg=read_json(CONFIG)
    new=ROOT/'results/full_natural_confirmation_v1'
    audit=read_json(new.parent/(new.name+'_audit.json'))
    if not audit['verified']:raise ValueError('Confirmation audit required.')
    old=ROOT/'results/full_depth_natural_v1'
    kept={f"coco/{r['source_image_id']}" for r in selected('old')}
    trials=read_json(old/'trials.json')
    with np.load(old/'scores.npz',allow_pickle=False) as z:scores=z['scores']
    before,before_rows=paired_metrics(trials,scores,cfg)
    after,after_rows=paired_metrics(trials,scores,cfg,kept)
    old_full,_=compute_metrics(trials,scores,config=cfg)
    mask=np.array([r['group_id'] in kept for r in trials])
    old_clean,_=compute_metrics([r for r,k in zip(trials,mask) if k],scores[mask],config=cfg)
    fresh=read_json(new/'metrics.json')
    from .common import csv_rows
    new_rows=csv_rows(new/'pairs.csv')
    a=np.array([r['correct'] for r in before_rows if r['template']==2 and r['order']==0],dtype=float)
    b=np.array([r['correct']=='True' for r in new_rows if r['template']=='2' and r['order']=='0'],dtype=float)
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    deltas=100*(b[rng.integers(len(b),size=(cfg['bootstrap_draws'],len(b)))].mean(1)-a[rng.integers(len(a),size=(cfg['bootstrap_draws'],len(a)))].mean(1))
    result={'old_full_pairs':before,'old_reviewed_pairs':after,'new_confirmation_pairs':fresh,
        'old_four_way_original':{'full':old_full['conditions']['t2_o0_original'],'reviewed':old_clean['conditions']['t2_o0_original']},
        'new_minus_old_primary_pp':{'delta':float(100*(b.mean()-a.mean())),'independent_bootstrap_ci95':np.quantile(deltas,[.025,.975]).tolist()},
        'review':{'old_total':200,'old_retained':len(kept),'old_excluded':200-len(kept),
            'new_candidates':len(read_json(SELECTION/'new_candidates.json')),
            'new_accepted':sum(r['accepted'] for r in read_json(SELECTION/'new_review.json')['decisions']),
            'new_selected':fresh['t2_o0']['n_sources'],'independent_human_reviewers':0}}
    out=ROOT/'reports/natural_confirmation_v1';out.mkdir(exist_ok=True)
    write_json(out/'metrics.json',result)
    write_csv(out/'old_review_decisions.csv',read_json(SELECTION/'old_review.json')['decisions'])
    write_csv(out/'new_review_decisions.csv',read_json(SELECTION/'new_review.json')['decisions'])
    table=[]
    for group,values in [('old_full',before),('old_reviewed',after),('new_confirmation',fresh)]:
        for condition,v in values.items():table.append({'dataset':group,'condition':condition,**v})
    write_csv(out/'all_conditions.csv',table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
    values=[d['t2_o0'] for d in [before,after,fresh]]
    y=np.array([v['accuracy_percent'] for v in values]);ci=np.array([v['wilson_ci95_percent'] for v in values]).T
    bars=ax.bar(range(3),y,color=['#929ca6','#74a2bd','#287a58'])
    ax.errorbar(range(3),y,yerr=[y-ci[0],ci[1]-y],fmt='none',color='black',capsize=4)
    ax.bar_label(bars,labels=[f"{v['correct']}/{v['n_sources']} ({v['accuracy_percent']:.2f}%)" for v in values],padding=35,fontsize=10)
    ax.axhline(50,color='#ae4444',linestyle='--',label='Hai cách ghép: mốc 50%')
    ax.set_xticks(range(3),['Cũ: đầy đủ','Cũ: sau tự rà soát','Mới: xác nhận định trước'])
    ax.set(ylabel='Ghép đúng 2×2 (%) và CI Wilson 95%',ylim=(0,90),title='T2 thuận — mỗi cặp là một ảnh gốc và bản lật')
    ax.legend(loc='upper left')
    fig.savefig(out/'confirmation.png',dpi=200,bbox_inches='tight');fig.savefig(out/'confirmation.pdf',bbox_inches='tight');plt.close(fig)
    def num(x):return f'{x:.2f}'.replace('.',',')
    def ci(v):return '['+'; '.join(num(x) for x in v)+']'
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Tập ảnh ngang & Số cặp & Ghép đúng (\%) & CI Wilson 95\% & Cả hai ảnh đúng (\%) \\',r'\midrule']
    for label,v in zip(['Cũ, đầy đủ','Cũ, sau tự rà soát','Mới, xác nhận'],values):
        lines.append(f"{label} & {v['n_sources']} & {num(v['accuracy_percent'])} & {ci(v['wilson_ci95_percent'])} & {num(v['both_images_correct_percent'])} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'primary_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Điều kiện mới & Đúng/tổng & Ghép đúng (\%) & CI Wilson 95\% & Cả hai ảnh đúng (\%) \\',r'\midrule']
    for key,v in fresh.items():
        label=f"T{key[1]} / "+('thuận' if key[-1]=='0' else 'đảo tương đương')
        lines.append(f"{label} & {v['correct']}/{v['n_sources']} & {num(v['accuracy_percent'])} & {ci(v['wilson_ci95_percent'])} & {num(v['both_images_correct_percent'])} "+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (out/'conditions_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    v=fresh['t2_o0'];f=result['old_four_way_original'];delta=result['new_minus_old_primary_pp']
    macros={'NewAcc':num(v['accuracy_percent']),'NewCI':ci(v['wilson_ci95_percent']),
        'NewP':f"{v['binomial_p_greater_0_5']:.4f}".replace('.',','),'NewCorrect':str(v['correct']),
        'NewBoth':num(v['both_images_correct_percent']),'NewTwoWay':num(v['two_way_accuracy_percent']),
        'OldCleanAcc':num(f['reviewed']['accuracy_percent']),'OldCleanCI':ci(f['reviewed']['accuracy_ci95_percent']),
        'OldCleanHorizontal':str(after['t2_o0']['n_sources']),'OldCleanPair':num(after['t2_o0']['accuracy_percent']),
        'DeltaAcc':num(delta['delta']),'DeltaCI':ci(delta['independent_bootstrap_ci95']),
        'NearTies':str(v['near_ties']),'ExactTies':str(v['exact_ties'])}
    (out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+val+'}' for k,val in macros.items())+'\n',encoding='utf-8')
    inputs=[CONFIG,ROOT/'docs/natural_confirmation_protocol.md',new/'manifest.json',new/'metrics.json',
        new.parent/(new.name+'_audit.json'),old/'manifest.json',old/'scores.npz',old/'trials.json',
        SELECTION/'old_review.json',SELECTION/'new_review.json',SELECTION/'selected.json',SELECTION/'freeze.json',
        ROOT/'vlm_probe/depth/confirmation.py',ROOT/'vlm_probe/depth/confirmation_report.py']
    write_json(out/'sources.json',{'input_sha256':{p.relative_to(ROOT).as_posix():sha256(p) for p in inputs},
        'derived_sha256':{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name not in ['sources.json','bao_cao_cap_nhat.tex','bao_cao_cap_nhat.pdf','README.md'] and p.suffix!='.log'}})
    print({'new_primary':v,'old_reviewed_four_way':f['reviewed'],'old_reviewed_pairs':after['t2_o0'],'delta':delta},flush=True)


if __name__=='__main__':run()
