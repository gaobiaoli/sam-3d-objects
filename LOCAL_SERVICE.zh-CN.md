# MySAM3d 本机 API 服务

项目克隆自 https://github.com/gaobiaoli/sam-3d-objects ，提交为
`3713a351ab335f89220ad0dd6045c084b1b15424`。
源码目录：`/home/bgao491/MySAM3d`。

## 访问

- 接口文档及交互测试页面：http://localhost:6006/docs
- OpenAPI：http://localhost:6006/openapi.json
- 三维推理：`POST /infer`
- bbox 分割：`POST /infer_sam`
- 结果下载：`GET /results/{request_id}/glb`、`/json`、`/mask`

服务运行在当前 WSL 中，监听 `0.0.0.0:6006`，已验证 WSL 和 Windows
宿主机通过 localhost 访问。公网入口和域名需要按实际部署环境另外配置。

## 服务管理

```bash
systemctl --user status mysam3d.service
systemctl --user start mysam3d.service
systemctl --user restart mysam3d.service
systemctl --user stop mysam3d.service
```

当前服务已启动，由用户级 systemd 托管，关闭启动终端后可以继续运行。
重启 WSL 后，可执行上面的 start 命令恢复服务。若希望随用户会话自动启动：

```bash
systemctl --user enable mysam3d.service
```

服务单元：`~/.config/systemd/user/mysam3d.service`。
启动脚本：`~/MySAM3d/run_service.sh`。首次启动会加载模型，需要约 4–5 分钟；
模型驻留后，每次请求不必重复加载。

查看实际应用日志：

```bash
tail -f ~/MySAM3d/outputs/service/server.log
```

## 环境与权重

服务复用 `sam3d` Conda 环境，并通过进程内的 `PYTHONPATH` 指定此项目源码。

- 三维权重：`~/Dataset/sam-3d-objects/checkpoints/pipeline.yaml`。
- 项目中的 `checkpoints/hf` 链接到上述权重目录。
- bbox 分割权重：`~/Dataset/segment-anything/sam_vit_h_4b8939.pth`。
- 已补装 Meta 官方 `segment_anything` 包，版本来源见
  `outputs/service/environment.json`。
- 输出 GLB、姿态 JSON 和生成掩码：`outputs/service/results/`。

这些路径由启动脚本设置，覆盖仓库中的 `/root/...` 默认路径。
`environments/local-pip-freeze.txt` 保存当前 Python 包版本。

## 已完成的测试

2026-11-06 已完成实际 HTTP 请求验证：

- Windows 和 WSL 的 `/docs`、`/openapi.json` 返回 HTTP 200。
- `/infer_sam` 使用真实 SAM ViT-H，从 bbox 生成有效的 1080×1080 PNG 掩码。
- `/infer` 使用仓库自带 RGB、bbox、深度图和相机内参，返回 HTTP 200。
- 单次推理约 16.23 秒，生成 8,129,464 字节 GLB，含 203,238 个顶点、406,390 个面。
- 已用 trimesh 重新读取 GLB，检查顶点和姿态数据有效，并验证 GLB/JSON/掩码下载。
- `python -m pytest -q`：1 passed；第三方库仍有弃用提示。
- `python -m pip check`：通过。

测试结果：`outputs/service/smoke-test/result.json`。
示例 GLB：`outputs/service/smoke-test/local_service_smoke.glb`。
HTTP 测试日志：`outputs/service/http-smoke.log`。
可重跑：

```bash
conda activate sam3d
cd ~/MySAM3d
python outputs/service/http_smoke.py
```

## 本机修复

仓库的 `/infer_sam` 原来使用 `FileResponse(content=...)` 返回内存中的 PNG，
会产生 HTTP 500。已改用支持字节内容的 `Response`，保留 PNG 类型和下载文件名。
回归测试 `tests/test_mask_response.py` 已先复现 500，再确认修复后返回可解码 PNG。

