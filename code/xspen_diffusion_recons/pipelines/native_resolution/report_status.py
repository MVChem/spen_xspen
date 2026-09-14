"""Generate a continuously refreshed campaign status and result index."""
import json
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def build_report():
    state = read_json(HERE/'campaign/status.json', {})
    preparation = read_json(HERE/'data/preparation_status.json', {})
    lines = ['# xSPEN 原生网格训练状态', '', f'更新：{datetime.now().astimezone().isoformat(timespec="seconds")}', '',
             f'队列状态：`{state.get("stage", "not_launched")}`；IXI 准备：{preparation.get("volumes", 0)}/1156 体，`{preparation.get("event", "pending")}`。', '',
             '两个几何协议来自同一个双chirp xSPEN序列族。四档是训练/仿真/输出网格；实采只有60×64和46×48，不能据输出像素数声称达到更高空间分辨率。', '',
             '| 模型 | 图像网格 | 状态 | 最新训练步 | 最佳EMA步 | 最佳验证损失 | 结果 |',
             '|---|---|---|---:|---:|---:|---|']
    results = []
    for job in state.get('jobs', []):
        run = HERE/'runs'/job['id']
        live = read_json(run/'status.json', {})
        vals = []
        if (run/'val_metrics.jsonl').exists():
            for row in (run/'val_metrics.jsonl').read_text().splitlines():
                try:
                    vals.append(json.loads(row))
                except json.JSONDecodeError:
                    pass
        baseline = read_json(run/'baseline.json', {})
        initial = baseline.get('initial_native_denoising_loss', float('inf'))
        best_row = min([dict(step=0, val_loss=initial), *vals], key=lambda row: row['val_loss'])
        best_val = best_row['val_loss']
        value = f'{best_val:.5f}' if best_val != float('inf') else '—'
        evaluation = Path(job.get('evaluation_out', HERE/'evaluation'/job['id']))
        report = read_json(evaluation/'summary.json')
        result_link = f'[结果]({evaluation.relative_to(HERE)}/summary.json)' if report else '待评价'
        step = live.get('step', vals[-1]['step'] if vals else 0)
        lines.append(f'| {job["id"]} | {job["shape"][0]}×{job["shape"][1]} | {job["stage"]} | {step} | {best_row["step"]} | {value} | {result_link} |')
        if report:
            results.append(dict(job=job['id'], shape=job['shape'], checkpoint_step=report['checkpoint_step'],
                                synthetic=report.get('synthetic'), real=report.get('real'), summary=str(evaluation/'summary.json')))
    lines += ['', f'每个模型{state.get("steps_per_model", 20000):,}步、batch{state.get("batch", 32)}；从旧人脑EMA初始化，固定462/58/58人的训练/验证/测试划分。最佳EMA可能来自step0，以验证集选择为准。', '',
              '完成后自动比较RO+RSS、Tikhonov、原生EDM+DiffPIR和旧128 prior。仿真仅在验证集选参数；真实数据没有干净真值，仅报告图像和观测残差。', '',
              '[实验说明](README.md) · [采集审计](audit/README.md) · [数据准备](PREPARATION.md) · [队列JSON](campaign/status.json)', '']
    temp = HERE/'STATUS.md.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(HERE/'STATUS.md')
    if results:
        temp = HERE/'results_summary.json.tmp'
        temp.write_text(json.dumps(dict(stage=state.get('stage'), results=results), indent=2)+'\n')
        temp.replace(HERE/'results_summary.json')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(build_report())
