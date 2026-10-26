"""Build a separate v2 report without overwriting the first-round evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    args = p.parse_args()
    root = args.results
    def load(name):
        return json.loads((root / name).read_text())
    table = load('retrieval_summary.json')
    quality = load('generation_quality.json')
    selection = load('calibration_summary.json')
    manifest = load('manifest.json')
    intervals = load('bootstrap_intervals.json')
    best = max(table, key=lambda r: r['real_test_mrr10'])
    lines = ['# 第二轮本地模型验证结果', '', '报告日期：2026-10-26。', '',
             '**性质：探索实验，不是独立确认或投稿结论。** 保留第一轮划分；测试集先前已查看。', '',
             '## 实际运行内容', '',
             f"- Qwen2.5-1.5B-Instruct，本地MPS、{manifest['generation']['dtype']}、固定版本、贪心解码。",
             '- 每个源回答生成零样本与3例真实问法引导两题，共512次生成。',
             '- 6种原词法系统 + BGE-small-zh-v1.5 + 固定RRF混合，共8种检索系统。',
             '- 3,746条回答库、750条真实校准池、2,994条真实测试问题；未将测试问句送入生成器。', '',
             '## 生成质量', '',
             f"每种方式尝试{quality['attempted_per_mode']}个源回答：零样本格式有效"
             f"{quality['valid_counts']['zero_shot']}，少样本格式有效{quality['valid_counts']['few_shot_3']}；"
             f"共同有效{quality['common_valid_sources']}个，用于配对比较。", '',
             '格式有效不等于问题能被源回答完整解答。未按任何候选检索器的召回结果过滤题目；'
             '原始失败记录保留在本地work目录。相关性为已知源回答关联，尚未人工审核。', '',
             '## 检索结果', '',
             '| 系统 | 真实 MRR@10 | 真实 Hit@5 | 同源关键词代理 | LLM零样本 MRR@10 | LLM少样本 MRR@10 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['system']} | {row['real_test_mrr10']:.4f} | {row['real_test_hit5']:.4f} | "
                     f"{row['keywords_matched_mrr10']:.4f} | {row['zero_shot_mrr10']:.4f} | {row['few_shot_3_mrr10']:.4f} |")
    lines += ['', f"真实测试中点估计最高的是 **{best['system']}**，MRR@10={best['real_test_mrr10']:.4f}。", '']
    winners = []
    for mode in ['keywords_matched', 'zero_shot', 'few_shot_3']:
        chosen = max(table, key=lambda r: r[mode + '_mrr10'])
        regret = best['real_test_mrr10'] - chosen['real_test_mrr10']
        winners.append({'mode': mode, 'selected': chosen['system'], 'regret': regret})
        lines.append(f"- {mode}选择{chosen['system']}，真实选择遗憾{regret:.4f}。")
    if all(w['selected'] == best['system'] for w in winners):
        lines += ['', '**本轮三种合成方式均选中点估计最佳系统，未观察到最佳系统反转。** '
                  '完整排序或绝对分数不同，仍不能自动证明选型有害。']
    else:
        lines += ['', '**至少一种合成方式的选择与真实点估计最优系统不同。** '
                  '必须结合差值区间、弱相关标签、重复数据及生成质量审计判断其意义，不能直接宣称普遍反转。']
    lines += ['', '## 同预算校准（全部预算和重复见JSON）', '',
              '3例演示包含在每组真实样本预算内；所有方法在同一真实校准子集上比较。'
              '表中比例是本轮五次抽样的结果，不是跨业务成功率。', '',
              '| 真实预算 | 生成方式 | 方法 | 平均 tau-b | 平均选择遗憾 | 选中点估计最佳比例 |',
              '|---:|---|---|---:|---:|---:|']
    for row in selection:
        if row['generation'] == 'keywords_matched' or row['method'] not in [
                'synthetic_uniform', 'synthetic_density_weighted', 'real_only', 'mix_real_density_0.5']:
            continue
        tau = '未定义' if row['mean_tau_b'] is None else f"{row['mean_tau_b']:.3f}"
        lines.append(f"| {row['budget']} | {row['generation']} | {row['method']} | {tau} | "
                     f"{row['mean_regret']:.4f} | {row['optimal_fraction']:.0%} |")
    lines += ['', '## 真实测试差值区间', '',
              '与BM25-char2的配对、逐项95%百分位bootstrap区间（1000次）。'
              '未做多重比较修正，不构成对新业务或因果收益的保证。', '',
              '| 系统 | MRR差值 | 差值区间 |', '|---|---:|---|']
    for row in intervals:
        lo, hi = row['paired_pointwise_delta_interval']
        lines.append(f"| {row['system']} | {row['delta_vs_bm25_char2']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    values = np.load(root / 'per_query_metrics.npz')['real_test'][:, :, 0]
    systems = manifest['systems']
    pair = values[:, systems.index('bge_small_zh')] - values[:, systems.index('rrf_bm25_bge')]
    rng = np.random.default_rng(20261027)
    boot_pair = np.array([pair[rng.integers(len(pair), size=len(pair))].mean() for _ in range(1000)])
    pair_record = {'comparison': 'BGE minus RRF', 'mean_delta': float(pair.mean()),
                   'paired_pointwise_95_interval': np.quantile(boot_pair, [0.025, 0.975]).tolist(),
                   'resampling_unit': 'question, descriptive, not independent-source confirmation'}
    (root / 'bge_vs_rrf.json').write_text(json.dumps(pair_record, indent=2))
    lo, hi = pair_record['paired_pointwise_95_interval']
    lines += ['', f"BGE与RRF直接配对比较：MRR@10差值{pair_record['mean_delta']:.4f}，"
              f"逐项95%描述性区间[{lo:.4f}, {hi:.4f}]。"]
    audit_path = root / 'semantic_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ['', '## 生成问题的源回答支持性初审', '',
                  '**这是助手进行的AI辅助来源一致性检查，不是独立人工或保险专家标注。** '
                  '从共同有效源回答中固定随机抽20个，每个检查两种问句，共40题；没有据此过滤本轮计分数据。', '',
                  '| 生成方式 | 源回答支持 | 部分支持 | 独立问句缺上下文 | 源回答不支持 |',
                  '|---|---:|---:|---:|---:|']
        for mode in ['zero_shot', 'few_shot_3']:
            counts = audit['counts'][mode]
            lines.append(f"| {mode} | {counts.get('supported',0)} | {counts.get('partial',0)} | "
                         f"{counts.get('needs_context',0)} | {counts.get('unsupported',0)} |")
        lines += ['', '初审发现的具体偏移包括：同等责任变为对方全责、医疗垫付条件变为住院后投保资格、'
                  '保留“这款/我的保险”却移除产品身份。源回答ID不能自动证明新问句的相关性标签有效。'
                  '这只是候选研究现象，样本小且审查并非独立，不能据此声称总体失败率或因果效应。']
    lines += ['', '## 成本与限制', '',
              f"- 本地生成耗时{manifest['generation']['elapsed_seconds']:.1f}秒，"
              f"真实问题和回答编码耗时{manifest['embedding']['elapsed_seconds']:.1f}秒；无付费API调用。",
              f"- BGE文档长度上限512token，截断{manifest['embedding']['truncated']['documents']}条回答；"
              '本轮未验证其他切片或长文档池化方式。',
              '- 只使用一个生成模型、一个密集模型、单一历史问题来源；公开语料可能与模型训练数据重叠，尚未排除。',
              '- 原始帖子URL/时间和完整使用许可链仍缺失；网友最佳回答并非经过专家核验的保险事实真值。',
              '- 未运行完整回答生成或Agent，未微调模型，未开展线上实验。',
              '- 正式创新仍需独立来源、专家标签、强基线与冻结后的确认实验。', '',
              '## 复现与审计', '',
              f"- 生成模型版本：`{manifest['generation']['revision']}`。",
              f"- 嵌入模型版本：`{manifest['embedding']['model_artifacts']['revision']}`。",
              '- 原6种词法检索器在真实数据上的逐题指标已与第一轮精确比对，全部一致。',
              '- 完整模型文件哈希、输入哈希、原始生成日志哈希、计分矩阵与逐次选择结果均保留。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (root / 'selection_outcomes.json').write_text(json.dumps(winners, indent=2), encoding='utf-8')
    draft_path = Path(__file__).resolve().parents[1] / 'paper/draft_zh.md'
    draft = draft_path.read_text(encoding='utf-8')
    heading = '### 5.2 本地LLM生成与密集检索验证'
    begin = draft.find(heading)
    end = draft.index('### 5.3') if '### 5.3' in draft else draft.index('## 6 局限与后续确认实验')
    if begin < 0:
        begin = end
    chosen_by_mode = {r['mode']: r['selected'] for r in winners}
    high_budget = {(r['generation'], r['method']): r for r in selection if r['budget'] == 200}
    real_fraction = high_budget['zero_shot', 'real_only']['optimal_fraction']
    zero_mix_fraction = high_budget['zero_shot', 'mix_real_density_0.5']['optimal_fraction']
    few_mix_fraction = high_budget['few_shot_3', 'mix_real_density_0.5']['optimal_fraction']
    audit_text = '源回答支持性初审尚未提供，不填写其质量结论。'
    if audit_path.exists():
        zero_counts = audit['counts']['zero_shot']
        few_counts = audit['counts']['few_shot_3']
        audit_text = (
            f"固定40题的AI辅助源回答支持性初审中，零样本{zero_counts.get('supported',0)}/20题被判为源回答支持，"
            f"少样本{few_counts.get('supported',0)}/20题被判为支持；少样本另有{few_counts.get('unsupported',0)}题"
            '出现源回答不支持的意图或条件变更。其余包含部分支持或独立问句缺上下文。'
            '初审不是独立人工或专家标签，没有用于过滤计分数据，不据此估计总体失败率或因果效应。')
    section = [heading, '',
               f"第二轮在{quality['common_valid_sources']}个共同格式有效源回答上比较关键词代理、"
               'Qwen零样本与3例真实问法引导生成。正式生成使用修复后的MPS float32/eager配置；'
               'float16异常运行保留但不纳入结果。', '',
               '| 系统 | 真实 MRR@10 | 匹配关键词代理 | LLM零样本 | LLM少样本 |',
               '|---|---:|---:|---:|---:|']
    for row in table:
        section.append(f"| {row['system']} | {row['real_test_mrr10']:.4f} | {row['keywords_matched_mrr10']:.4f} | "
                       f"{row['zero_shot_mrr10']:.4f} | {row['few_shot_3_mrr10']:.4f} |")
    section += ['',
                f"真实测试点估计最高为{best['system']}。零样本选择{chosen_by_mode['zero_shot']}，"
                f"少样本选择{chosen_by_mode['few_shot_3']}；"
                f"BGE相对RRF的真实MRR差值为{pair_record['mean_delta']:.4f}，"
                f"问题级配对bootstrap逐项95%描述性区间[{lo:.4f}, {hi:.4f}]。"
                f"关键词代理则选择{chosen_by_mode['keywords_matched']}。该结果提示不同合成方式可能影响选型，"
                '但仍受单源弱标签与生成质量限制，不是独立业务确认。', '',
                f"预算200时，真实样本基线选中点估计最佳系统的比例为{real_fraction:.0%}；"
                f"零样本固定混合为{zero_mix_fraction:.0%}，少样本固定混合为{few_mix_fraction:.0%}。"
                '各方法的完整结果包括负结果，不能仅凭局部排名一致性提升声称选型收益。', '',
                audit_text, '',
                '完整生成质量、360组校准比较、逐题计分与失败记录见第二轮实验报告。', '',]
    draft_path.write_text(draft[:begin] + '\n'.join(section) + draft[end:], encoding='utf-8')
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, mode, label in [(-0.27, 'real_test', 'Held-out user questions'),
                                (-0.09, 'keywords_matched', 'Matched keyword proxy'),
                                (0.09, 'zero_shot', 'Qwen zero-shot'),
                                (0.27, 'few_shot_3', 'Qwen 3 demonstrations')]:
        ax.bar(x + offset, [r[mode + '_mrr10'] for r in table], 0.18, label=label)
    ax.set_xticks(x, [r['system'].replace('_', '\n') for r in table], fontsize=8)
    ax.set_ylabel('MRR@10 (weak relevance labels)')
    ax.set_ylim(0, 1)
    ax.set_title('Local model validation: exploratory, one historical source')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'comparison.png', dpi=180)
    plt.close(fig)
    print('V2 report and figure written from observed results.')


if __name__ == '__main__':
    main()
