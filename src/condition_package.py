"""Publish aggregate proxies and genuinely blank, method-blinded review packs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from condition_guard import condition_check
from experiment import write_json

HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>保险条款研究独立标注</title><style>body{font:16px/1.7 system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 24px;color:#182431;background:#f5f7fa}article{background:white;border:1px solid #d7dfe8;border-radius:12px;padding:24px;margin:18px 0}button,select,textarea{font:inherit;padding:8px;margin:6px}button{cursor:pointer}textarea{display:block;width:94%;height:90px}#question,#reference{white-space:pre-wrap}small{color:#536477}.label{font-weight:700}a{color:#135aba}</style>
<h1>保险条款研究标注 · __REVIEWER__</h1><p>仅核对给定资料版本中的等待期问题。本页没有自动参考标签或生成方法名称。请独立核验原PDF后填写；不要把“不因等待期排除”理解为一定赔付。</p>
<p><select id="task"><option value="action">场景判断（80题）</option><option value="fidelity">生成问题条件保持（40题）</option></select><span id="progress"></span></p>
<article><div id="identifier" class="label"></div><p id="context"></p><p><a id="source" target="_blank" rel="noopener">打开官方PDF证据页</a></p><div id="refwrap"><div class="label">改写前问题</div><p id="reference"></p></div><div id="qtitle" class="label"></div><p id="question"></p><label>您的独立判断 <select id="label"></select></label><textarea id="reason" placeholder="依据、缺失条件或规则问题；建议写明条款位置"></textarea><small>更改自动保存在当前浏览器本地，不向服务器发送。请每位审核者使用各自文件与浏览器资料。</small></article>
<button id="prev">上一题</button><button id="next">下一题</button><button id="export">导出本人标注JSON</button><p id="saved"></p>
<script>const DATA=__PAYLOAD__;const reviewer="__REVIEWER__";const key="condition-review-"+reviewer;const state=JSON.parse(localStorage.getItem(key)||"{}");let index=0;const el=id=>document.getElementById(id);const options={action:["","WAIT_EXCLUDED","WAIT_NOT_EXCLUDED","CLARIFY","SOURCE_INSUFFICIENT"],fidelity:["","PRESERVED","NOT_PRESERVED","UNCERTAIN"]};
function items(){return DATA.filter(x=>x.task===el("task").value)}function persist(){const row=items()[index];state[row.id]={id:row.id,task:row.task,label:el("label").value,reason:el("reason").value};localStorage.setItem(key,JSON.stringify(state));el("saved").textContent="当前浏览器本地已保存；尚未导出提交。"}
function show(){const rows=items();index=Math.max(0,Math.min(index,rows.length-1));const row=rows[index];const old=state[row.id]||{};el("identifier").textContent=row.id;el("progress").textContent=` ${index+1}/${rows.length}，已填 ${rows.filter(r=>state[r.id]&&state[r.id].label).length}`;el("context").textContent=row.context;el("source").href=row.source_url+"#page="+row.page;el("source").textContent=`官方PDF · 第${row.page}页（请核对完整条款）`;el("refwrap").hidden=row.task!=="fidelity";el("reference").textContent=row.reference_question||"";el("qtitle").textContent=row.task==="action"?"待判断问题":"待审核改写问题";el("question").textContent=row.question;el("label").replaceChildren(...options[row.task].map(v=>{const o=document.createElement("option");o.value=v;o.textContent=v||"请选择，不预填";return o}));el("label").value=old.label||"";el("reason").value=old.reason||""}
el("prev").onclick=()=>{persist();index--;show()};el("next").onclick=()=>{persist();index++;show()};el("task").onchange=()=>{index=0;show()};el("label").onchange=persist;el("reason").oninput=persist;el("export").onclick=()=>{persist();const blob=new Blob([JSON.stringify({reviewer,records:Object.values(state)},null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=reviewer+"_annotations.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};show();</script></html>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--generation',type=Path,required=True)
    args=p.parse_args();root=args.root;out=args.data
    sources=json.loads((out/'sources.json').read_text());source_by_id={r['id']:r for r in sources}
    cases=[json.loads(s) for s in (out/'controlled_cases.jsonl').read_text().splitlines()]
    inputs=json.loads((root/'generation_inputs.json').read_text());seeds={r['seed_id']:r for r in inputs['jobs']}
    raw=[json.loads(s) for s in (args.generation/'generations.jsonl').read_text().splitlines()]
    if len(raw)!=40 or len({(r['seed_id'],r['method']) for r in raw})!=40:
        raise ValueError('Expected exactly 40 unique paired generations')
    if not (args.generation/'manifest.json').exists():raise ValueError('Generation did not finish')
    checked=[]
    for row in raw:
        seed=seeds[row['seed_id']]
        revised=condition_check(row['question'],seed['frame'],seed['required_fields'])
        checked.append({**row,**revised,'guard_revision':'v1: distinguishes first diagnosis from first enrollment; resolves negative accident cause'})
    with (out/'generated_questions.jsonl').open('w',encoding='utf-8') as f:
        for row in checked:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={}
    for method in ['ordinary_style','condition_prompt']:
        rows=[r for r in checked if r['method']==method]
        summary[method]={'attempts':len(rows),'format_valid':sum(r['format_valid'] for r in rows),
                         'all_required_fields_proxy_preserved':sum(r['format_valid'] and r['proxy_all_required_fields_preserved'] for r in rows),
                         'flagged_or_unresolved':sum(not r['format_valid'] or not r['proxy_all_required_fields_preserved'] for r in rows),
                         'field_flags':{field:sum(field in r['mismatched_or_unresolved_fields'] for r in rows)
                                        for field in ['policy_id','enrollment','policy_day','cause','plan_wait_days','approved_waiver']}}
    write_json(out/'generation_proxy_summary.json',summary)
    review_dir=out/'independent_review';review_dir.mkdir(exist_ok=True)
    blind=[];mapping={}
    for i,row in enumerate(cases):
        source=source_by_id[row['source_id']];key=f'A{i+1:03d}'
        blind.append(dict(id=key,task='action',question=row['question'],context=row['task_context'],
                          source_url=source['url'],page=source['anchor']['pdf_page_1based']))
        mapping[key]={'case_id':row['id'],'reference_label':row['reference_label']}
    for i,row in enumerate(checked):
        source=source_by_id[row['source_id']];key=f'F{i+1:03d}'
        blind.append(dict(id=key,task='fidelity',question=row['question'],reference_question=seeds[row['seed_id']]['seed_question'],
                          context=inputs['task_context'],source_url=source['url'],page=source['anchor']['pdf_page_1based']))
        mapping[key]={'seed_id':row['seed_id'],'method':row['method']}
    write_json(root/'blind_review_mapping_private.json',mapping)
    for reviewer,seed in [('reviewer_A',20261029),('reviewer_B',20261030)]:
        order=np.random.default_rng(seed).permutation(len(blind));items=[blind[i] for i in order]
        payload=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
        (review_dir/(reviewer+'.html')).write_text(HTML.replace('__PAYLOAD__',payload).replace('__REVIEWER__',reviewer))
        with (review_dir/(reviewer+'_blank.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            columns=['id','task','question','reference_question','context','source_url','page','label','reason']
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
            for item in items:w.writerow({k:item.get(k,'') for k in columns})
    generation_manifest=json.loads((args.generation/'manifest.json').read_text())
    write_json(out/'collection_summary.json',{'date':'2026-10-26','formal_documents':len(sources),
        'controlled_cases':len(cases),'all_cases_are_synthetic':True,'seed_groups':len(seeds),
        'raw_generation_count':len(raw),'generation':generation_manifest,'proxy_metrics':summary,
        'independent_review_status':'PENDING: blank packs only; no human labels or agreement scores exist',
        'guard_script_sha256':hashlib.sha256((Path(__file__).with_name('condition_guard.py')).read_bytes()).hexdigest(),
        'package_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    lines=['# 可追溯条款与条件守恒：首批数据及本地验证','', '日期：2026-10-26。', '',
           '**状态：数据与自动代理实验完成，独立人工标注尚未完成。** 所有80条场景是受控合成；不能称真实用户日志或独立专家测试集。','',
           f'本批包含{len(sources)}份正式条款、3个保险公司集团、20个场景组与80条问题。另保留1份产品说明书供来源审计，不作为正式条款；错误PDF链接已排除。','',
           '## 正式条款来源','', '| 产品 | 证据PDF页码 | 官方来源 |','|---|---:|---|']
    for source in sources:
        lines.append(f"| {source['title']} | {source['anchor']['pdf_page_1based']} | [官方PDF]({source['url']}) |")
    lines += ['', 'URL、下载时间、完整文件SHA256、规范化字符区间、证据片段哈希及PDF坐标均已保存。'
              '原始PDF及完整证据文本只在work目录保留，未随数据包再发布。', '',
              '## 规则与问题构造', '',
              '每组含完整、表面改写、删除关键条件和单条件反事实四种问题。20条缺失问题各有两个导致不同判断的补全见证；'
              '20条反事实均实际改变有限规则世界中的参考判断。规则参考由助手整理，仍可能误读条款，须由独立人员核验。', '',
              '只判断等待期这一项排除，假定其他条件均满足；不判断真实赔付资格。按月等待期没有被粗略折成天数。', '',
              '## 本地模型对照','', '| 改写方式 | 尝试 | 格式有效 | 所有必需字段代理检查通过 | 标记或待核实 |',
              '|---|---:|---:|---:|---:|']
    for method,row in summary.items():
        lines.append(f"| {method} | {row['attempts']} | {row['format_valid']} | {row['all_required_fields_proxy_preserved']} | {row['flagged_or_unresolved']} |")
    lines += ['', '以上是保守字符串/数字槽位检查，可能漏认同义表达，也可能误认词面提及；不是语义准确率、专家通过率或论文方法收益。'
              '两种方法均保留全部生成结果，没有用候选检索器召回过滤，没有靠丢弃失败来增加样本数。', '',
              '### 逐字段保守代理标记明细（v1 严格槽位检查）', '',
              '| 改写方式 | 产品全名 (`policy_id`) | 投保状态 (`enrollment`) | 生效天数 (`policy_day`) | 事故原因 (`cause`) | 计划等待期 (`plan_wait_days`) | 批准免等待期 (`approved_waiver`) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for method, row in summary.items():
        ff = row['field_flags']
        lines.append(f"| {method} | {ff['policy_id']} | {ff['enrollment']} | {ff['policy_day']} | {ff['cause']} | {ff['plan_wait_days']} | {ff['approved_waiver']} |")
    diag_path = out / 'condition_drift_diagnostics.json'
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding='utf-8'))
        lines += ['', '### 表面措辞变体与实质条件漂移诊断（`analyze_v3.py`）', '',
                  '| 改写方式 | v1严格槽位通过 | 表面同义归一槽位通过 | v1因表面句式误拒数 | 独立问句保持原规则结论 | 退化为需追问 (`CLARIFY`) | 条件数值被改写 | 发生反相标签碰撞 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for method in ['ordinary_style', 'condition_prompt']:
            m = diag['methods'][method]
            sd = m['standalone_decision_outcomes']
            col = sum(1 for c in diag['collisions'] if c['method'] == method and c['opposite_label_collision'])
            lines.append(f"| {method} | {m['strict_v1_proxy_pass']}/20 | {m['surface_normalized_proxy_pass']}/20 | "
                         f"{m['v1_false_negative_on_surface_phrasing']} | {sd['PRESERVED_DECISION']}/20 | "
                         f"{sd['DEGRADED_TO_CLARIFY']}/20 | {sd['CONDITION_MUTATED']}/20 | {col}组 |")
        lines += ['', '注：`ordinary_style` 在 `condition-009`（7天等待期计划、第10天出险，参考结论 `WAIT_NOT_EXCLUDED`）与 `condition-010`（25天等待期计划、第10天出险，参考结论 `WAIT_EXCLUDED`）上由于同时丢弃计划等待期天数，生成了逐字完全相同的问句（反相标签碰撞）；并在 `condition-013/015/016` 中将“生效第20天”改写为“生效后的第二天”。`condition_prompt` 在表面同义归一后有 19/20 题完整保留全部业务条件（仅 `condition-008` 遗漏产品名称），而 v1 严格正则对“保单生效是第20天”“同意免去/免除等待期”等 9 处同义句式产生了保守误拒，印证了自动槽位检查只能作为保守初筛而非语义金标准。']
    lines += ['', f"生成模型为本地Qwen2.5-1.5B-Instruct，float32/eager，40次生成耗时{generation_manifest['elapsed_seconds']:.1f}秒。", '',
              '## 独立标注包', '',
              'independent_review目录中有A/B两份离线HTML和空白CSV，各含80条场景判断与40条生成一致性审核。'
              '没有预填标签，也没有显示生成方法名。每位审核人只用自己的文件独立填写，导出后再做分歧统计与裁决。', '',
              '**当前没有完成的独立人工标签，因此不能报告标注一致率、人工确认提升或独立测试效果。**', '',
              '## 下一步判定标准', '',
              '先核验规则与40条生成问句的真实条件保持，再决定是否扩大到更多公司和任务。普通提示与显式保留提示的对照仅用于机制探索；'
              '已有一致性、可回答性与反事实研究，当前不声称算法新颖或具备投稿终稿证据。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print('Blank review packs written; human annotation remains pending.')


if __name__=='__main__':main()
