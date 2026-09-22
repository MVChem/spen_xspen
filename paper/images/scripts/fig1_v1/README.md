# fig1_v1 可编辑重建

输出：`../../ppt/fig1_v1.pptx`；`../../ppt/fig1_v1.png` 为最终 PPT 经 LibreOffice 渲染的预览。

## 重新生成

依赖 Python 3、python-pptx、Pillow；渲染另需 LibreOffice 和 Poppler（pdftoppm）。

```sh
python -m pip install python-pptx Pillow
python build.py
libreoffice --headless --convert-to pdf --outdir /tmp/fig1_render ../../ppt/fig1_v1.pptx
pdftoppm -scale-to 1586 -png -singlefile /tmp/fig1_render/fig1_v1.pdf ../../ppt/fig1_v1
```

从任意目录运行 build.py 均可；脚本通过自身位置定位随附素材。整体移动输出目录后仍可重建。

## 编辑能力与还原范围

- 单页，页面比例与 1586 × 992 参考图一致，173 个有独立名称的对象。
- 标题、说明和公式为原生文本框，可直接改字；公式使用文字及上下标，不是 Office 公式对象。默认文字 Arial，数学文字使用 Liberation Serif；原图无可用字体元数据，字体与公式排版为近似匹配。
- 流程框、U-Net 示意、连线、箭头、频谱曲线、采样行均为原生形状或自由曲线，可修改颜色、位置及顶点。箭头使用路径线端箭头，箭身与箭头为同一对象。
- 每支箭头独立，每个文本框独立。通过 PowerPoint「选择窗格」按名称操作；单独移动框不会自动移动其文字及相邻连线，可按需多选。避免整页大分组，便于局部替换。
- MRI、噪声、复数线圈数据缩略图、相位彩图和色标、采集网格及 SPEN 矩阵为原图裁切位图，不能编辑内部像素为形状。原始图片仅放在 assets 中供脚本裁切参考，未整页贴入 PPT。
- 无嵌入 SVG。RO 频谱为视觉近似的可编辑示意曲线，不是恢复的科研数据；色块用平色近似原图淡渐变。公式分式采用行内除号表达，保留含义。
- 预览已从最终 PPT 导出并核对布局；素材保持原图分辨率。构建脚本会覆盖生成结果，修改 PPT 后应另存，避免手工修改被脚本覆盖。
