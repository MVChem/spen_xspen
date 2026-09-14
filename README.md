# SPEN / xSPEN 重建

SPEN 与 xSPEN diffusion reconstruction 的统一工作项目。助手工作时遵循 [项目约束](AGENTS.md)。

| 目录 | 用途 |
|---|---|
| [code/](code/README.md) | 各个代码项目及其源码，每个项目可以有自己的 `runs/`，记录实验及运行产物 |
| [experiments/](experiments/) | 用户从各项目 `runs/` 中挑选的重要结果，按文件夹展示，由用户自行放入 |
| [notes/](notes/) | 用户需要保留的笔记，目前以 Markdown 为主；旧内容已清空 |
| [tmp/](tmp/) | 当前实验进展、临时 notes 和展示材料；用户查看后决定是否移到其他位置 |
| [paper/](paper/README.md) | 论文源码、参考文献与正式配图 |
| [周报/](周报/) | 周报及相关材料 |

`tmp/` 一般在每次开始工作时为空，由用户手动清理，助手不自动删除其中的内容。

notes 名称与 `experiments/` 下的实验文件夹名称均以六位日期 `YYMMDD` 结尾，例如 `重建观察_260914.md`、`重建对比_260914/`，其中 `260914` 表示 2026 年 9 月 14 日。

```text
code/<项目>/              项目源码
code/<项目>/runs/         该项目的实验记录与产物
tmp/                      当前进展和供查看的临时材料
notes/<主题>_YYMMDD.md     保留的笔记
experiments/<主题>_YYMMDD/ 用户选取的重要实验结果
```
