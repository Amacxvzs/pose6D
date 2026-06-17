# pose6d 数据采集与相机标定工具

这个文件夹用于论文实验的第一阶段：采集 Astra Pro Plus 的 RGB/深度数据，并完成相机内参标定。

## 1. PyCharm 打开方式

1. 在 PyCharm 中选择 `Open`。
2. 打开文件夹：`D:\1.sjcl\pose6d`。
3. Python 解释器建议选择：`D:\PY\python.exe`。
4. 依赖当前已检测到 `opencv-python` 和 `numpy` 可用；如新环境缺依赖，运行：

```powershell
pip install -r requirements.txt
```

## 2. 检查相机编号

```powershell
python scripts/check_cameras.py --preview
```

如果打开的不是 Astra Pro Plus 彩色相机，试：

```powershell
python scripts/check_cameras.py --max-index 10
```

然后把 `configs/capture_config.json` 和 `configs/calibration_config.json` 里的 `color_camera_index` 改成正确编号。

## 3. 采集论文数据集

手动采集：

```powershell
python scripts/capture_dataset.py --class-name bolt --split train --scene stack
```

快捷键：

- `Space`：保存一帧
- `a`：自动保存开关
- `q` 或 `Esc`：退出

自动采集，每 1 秒保存一帧：

```powershell
python scripts/capture_dataset.py --class-name bolt --split train --scene stack --auto --interval 1.0
```

输出目录：

```text
data/raw/rgb/train/bolt/
data/raw/depth/train/bolt/
data/raw/preview/train/bolt/
```

说明：如果 Python OpenCV 没有成功打开 OpenNI2 深度流，脚本会自动退化为只采 RGB。RGB 数据仍然可以先用于 YOLO 检测和角点标注。

当前本机测试结果：OpenCV 可以打开 Astra Pro Plus 彩色相机 `index=0`，但 Python OpenCV 暂时打不开 OpenNI2 深度流，因此会先按 RGB-only 采集。

## 4. 采集棋盘格标定图

默认棋盘格为 `9 x 6` 个内角点，方格边长 `25 mm`。如果你的棋盘格不同，修改：

```text
configs/calibration_config.json
```

启动采集：

```powershell
python scripts/capture_calibration.py
```

建议采集 `20-30` 张，角度要丰富：

- 正对相机
- 左右倾斜
- 上下倾斜
- 远近变化
- 图像四角都出现棋盘格

快捷键：

- 检测到棋盘格后按 `Space` 保存
- `q` 或 `Esc` 退出

## 5. 计算相机内参

```powershell
python scripts/calibrate_camera.py
```

输出：

```text
data/calibration/camera_intrinsic.json
data/calibration/camera_intrinsic.yaml
```

后续 EPnP 需要用这里的：

- `fx`
- `fy`
- `cx`
- `cy`
- 畸变系数

## 6. 建议的第一批实验数据

先不要一次采 7200 张。第一轮建议：

| 类别 | train | val | test |
|---|---:|---:|---:|
| bolt | 240 | 30 | 30 |
| nut | 240 | 30 | 30 |
| washer | 240 | 30 | 30 |

第一轮约 900 张，够先跑通检测、角点预测、EPnP 和论文里的初步实验表。
