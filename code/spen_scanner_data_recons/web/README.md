# SPEN 重建工作台

FastAPI + React 的只读浏览界面，直接使用现有 runs/all_raw_260915 数据。
白色简洁布局参考 Figma 模板目录中的 [Simple dashboard mockup](https://www.figma.com/community/file/1048559673287861086/dashboard/)；社区节点未能直接读取，当前实现为原创布局，并非模板导出或逐像素复刻。

## 启动

在 code/spen_scanner_data_recons 下执行：

```bash
python -m pip install -r web/requirements.txt
cd web/frontend
npm ci --include=dev
npm run build
cd ../..
python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 8765
```

本工作区可使用 `/home/data2/chk/workspace/2026/.venv/bin/python`。打开 http://127.0.0.1:8765 。远程机器可通过 SSH 转发端口：`ssh -L 8765:127.0.0.1:8765 <服务器>`。

开发时后端保持启动，在 web/frontend 执行 `npm run dev`，Vite 将 /api 和 /data 转发至 8765。
通过环境变量 SPEN_RUN 指定其他已由旧流程生成、校验的运行目录。服务启动时复用 scripts/build_viewer.py 的 prepare_payload，核对 PNG 索引、状态、坐标和文件引用；只在启动时读取清单，数据更新后重启服务。

- GET /api/catalog：汇总统计与扫描索引，不包含逐帧数据。
- GET /api/experiments/{experiment}：指定实验的完整扫描和帧。
- /data/：所选运行目录的静态文件（PNG、NPZ、旧图集等）。

实验和显示模式记录在 URL hash 中；图像按需加载；实验切换取消旧请求。保留零帧扫描、质量警告、独立设窗说明和未校正状态。旧离线 HTML 浏览器继续可用。新版需要运行服务，不支持直接用 file:// 打开。

服务默认仅监听本机，无账号系统；/data 提供整个所选运行目录，请在需要远程访问时使用 SSH 隧道。
