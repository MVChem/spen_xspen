# 项目数据

[鼠类 MRI 数据与溯源](rodent_mri/README_260914.md) 收录实验室原始采集文件及公开数据的下载原件。

[192×192 鼠脑训练数据入口](rodent192_training_260915/README.md) 通过软链接提供本轮训练使用的全部 10,633 张 PNG、实际训练数组及逐图来源清单。

[真实 SPEN / xSPEN 采集数据](spen_acquired_260915/README.md) 保存 45 组成像原始实验（751 个非空 SPEN/xSPEN 扫描）、2 组波谱实验和 501 份配套 MAT 的实体副本；另存 5 份待核实来源的历史 MAT 候选。附实验索引、逐扫描对应关系和 SHA256 清单，本轮已测试的 64 份 MAT 在清单中单独标记。

其他对比度也已接入：高分辨率入口为 [Tc1/WT 离体 GRE，55 份、标称 40 μm](rodent_mri/public_rodent_mri/figshare_tc1_wt_3258139/) 和 [ds004644 离体 FLASH，22 份、约 40 μm](rodent_mri/public_rodent_mri/ds004644/)。后者还保留活体 MP2RAGE、UTE；[Aging](rodent_mri/public_rodent_mri/figshare_aging_28433102/) 保留已下载 DWI。具体对比度、体素与文件位置见 [公开数据说明](rodent_mri/public_rodent_mri/README_260914.md)。

部分本机已有数据通过符号链接接入；旧服务器采集文件及本轮 SPEN 采集数据以实体副本保存。来源、原路径、文件大小和 SHA256 随数据保存。各入口的存储方式见对应 README；使用时将输出放在相应代码项目的 `runs/`，不要覆盖输入数据。

`runs/` 保存导入日志。
