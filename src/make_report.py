"""Render only observed results; never fill in unrun experimental claims."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'


def load(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def main():
    audit = load('audit.json')
    manifest = load('manifest.json')
    table = load('retrieval_summary.json')
    calibration = load('calibration_summary.json')
    best_real = max(table, key=lambda x: x['real_test_mrr10'])
    best_syn = max(table, key=lambda x: x['synthetic_keywords_mrr10'])
    best_prefix = max(table, key=lambda x: x['source_prefix_diagnostic_mrr10'])
    prefix_regret = best_real['real_test_mrr10'] - best_prefix['real_test_mrr10']
    lines = [
        '# 首轮探索实验结果', '',
        '日期：2026-10-25。仅为单源、词法检索、弱相关标签的探索实验。', '',
        f"原始{audit['raw_rows']}条记录，经筛选和归并后有{audit['question_groups']}个问题、"
        f"{audit['documents']}个回答；校准池{audit['calibration_pool']}题，真实测试{audit['test_questions']}题。",
        f"问题按{audit['connected_groups']}个关联组划分，跨划分共享回答为0；"
        f"跨划分标题最大字符余弦相似度为{audit['max_cross_split_title_cosine']:.4f}。", '',
        '“真实”仅指发布者声明问题来自用户，并非独立核验的原始业务日志。', '',
        '| 检索器 | 真实测试 MRR@10 | 真实 Hit@5 | 关键词代理 MRR@10 | 回答前缀诊断 MRR@10 |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in table:
        lines.append(f"| {row['system']} | {row['real_test_mrr10']:.4f} | {row['real_test_hit5']:.4f} | "
                     f"{row['synthetic_keywords_mrr10']:.4f} | {row['source_prefix_diagnostic_mrr10']:.4f} |")
    lines += ['',
        f"真实测试和关键词代理都选择了 **{best_real['system']}**。因此，虽然绝对分数与完整排序不同，"
        '这一对照没有出现最佳系统选错，不能声称已证实关键词合成导致错误选型。', '',
        f"回答前缀诊断选择 **{best_prefix['system']}**，其真实测试MRR@10比最佳系统低{prefix_regret:.4f}。"
        '这是刻意复制源文档的诊断对照，不能当作普通LLM问句生成的实证结论。', '',
        '## 校准实验', '',
        f"每个预算重复{manifest['repeats']}次。tau-b衡量完整排序；遗憾衡量最佳系统选择。", '',
        '| 真实问题预算 | 方法 | 平均 tau-b | 平均选择遗憾 | 选中最佳比例 |',
        '|---:|---|---:|---:|---:|',
    ]
    for row in calibration:
        tau = f"{row['mean_tau_b']:.3f}" if row['mean_tau_b'] is not None else '未定义'
        lines.append(f"| {row['budget']} | {row['method']} | {tau} | {row['mean_regret']:.4f} | "
                     f"{row['optimal_selection_fraction']:.0%} |")
    lines += ['',
        '未观察到密度校准相对于均匀关键词代理的最佳系统选择收益；关键词代理本身已选中最佳。'
        '本轮预算50及以上的真实样本基线均选中最佳，并在较大预算上恢复了更接近真实测试的完整排序。'
        '不能把预算25时固定混合的个别优势解释成普遍显著提升。', '',
        '## 结论边界', '',
        '- 不同代理形式可改变排名，但合成分数偏高本身不能证明选型错误。',
        '- 当前密度重加权没有展现可发表的方法收益；应保留负结果。',
        '- 源回答标签不是穷尽相关性判断；关键词代理的题目可能不自然。',
        '- 未运行LLM生成、密集检索、重排、完整RAG/Agent或线上测试。',
        '- 重复采样共享测试集，未进行独立来源确认，不能据此声称统计显著或部署收益。', '',
        '## 复现', '',
        f"- 随机种子：{manifest['seed']}；运行耗时：{manifest['elapsed_seconds']:.2f}秒。",
        f"- 数据 SHA256：`{manifest['data_sha256']}`。",
        f"- 运行代码 SHA256：`{manifest['code_sha256']}`。",
        '- 完整数值见同目录 JSON 和 per_query_metrics.npz。',
    ]
    report = '\n'.join(lines) + '\n'
    (RESULTS / 'report.md').write_text(report, encoding='utf-8')
    draft_path = ROOT / 'paper/draft_zh.md'
    draft = draft_path.read_text(encoding='utf-8')
    start = draft.index('## 5 探索结果')
    end = draft.index('### 5.2') if '### 5.2' in draft else draft.index('## 6 局限与后续确认实验')
    section = ['## 5 探索结果', '',
        f"首轮从{audit['raw_rows']}条原始记录构建{audit['documents']}条回答库，"
        f"包含{audit['calibration_pool']}条校准池问题与{audit['test_questions']}条真实测试问题。", '',
        '| 检索器 | 真实 MRR@10 | 关键词代理 MRR@10 | 前缀诊断 MRR@10 |',
        '|---|---:|---:|---:|',
    ]
    for row in table:
        section.append(f"| {row['system']} | {row['real_test_mrr10']:.4f} | "
                       f"{row['synthetic_keywords_mrr10']:.4f} | {row['source_prefix_diagnostic_mrr10']:.4f} |")
    section += ['',
        f"关键词代理与真实测试均选择{best_real['system']}，分别得到"
        f"{best_syn['synthetic_keywords_mrr10']:.4f}与{best_real['real_test_mrr10']:.4f}的MRR@10。"
        '这种绝对分数差异不能直接等同于错误选型；题目语义与标签机制也不同。', '',
        f"复制回答前缀的诊断对照选择{best_prefix['system']}，对应真实选择遗憾为{prefix_regret:.4f}。"
        '该对照旨在暴露源文档措辞敏感性，不能用于推断普通LLM合成方法的表现。', '',
        '密度比校准没有改善关键词代理的最佳系统选择，因为该代理已选中最佳系统。'
        '预算50、100和200时，真实样本基线在五次重复中均选中最佳系统。'
        '这一结果不支持当前校准组合具有稳定方法优势；完整排序、预算与重复结果见实验报告。', '',
        '以上结果为探索性结果。候选支持审计与暂缓选型机制尚未验证，'
        '必须在新增系统和独立来源上开展确认实验，不能在本测试上反复调整并报告为独立验证。', '',
    ]
    draft_path.write_text(draft[:start] + '\n'.join(section) + draft[end:], encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        positions = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for offset, field, label in [(-0.25, 'real_test_mrr10', 'Held-out user questions'),
                                     (0, 'synthetic_keywords_mrr10', 'Extractive keyword proxy'),
                                     (0.25, 'source_prefix_diagnostic_mrr10', 'Source-prefix diagnostic')]:
            ax.bar(positions + offset, [r[field] for r in table], width=0.25, label=label)
        ax.set_xticks(positions, [r['system'].replace('_', '\n') for r in table], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('MRR@10 (weak relevance labels)')
        ax.set_title('Exploratory Chinese insurance answer retrieval')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / 'retrieval_comparison.png', dpi=180)
        plt.close(fig)
    except ImportError:
        print('Optional plot skipped: matplotlib not installed')
    print('Report and draft results updated from observed JSON.')


if __name__ == '__main__':
    main()
