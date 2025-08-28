#from scripts.class_frame_extractor import VideoFrameExtractor 导入
#extractor = VideoFrameExtractor(output_dir="dataset_frames") 定义路径
#frames = extractor.extract(
#    "D:\MachineLearning\Cholecyst-and-Instrument-Segmentation-Model\scripts\GmUqWJFFlx08qWpUnRTO01041201Zi810E010.mp4",
#    fps=2,            每秒取 2 帧
#    start=10,         从第 10 秒开始
#    end=60,           到第 60 秒结束
#    size=(512, 512),  输出统一大小
#    fmt="jpg",        图像格式
#    jpg_quality=90,   图像质量
#    batch_size=30,    包大小
#    mode=2，          选择模式，mode=1为ffmpeg mode=2为opencv 只有opencv可以分包
#)
# print(f"共提取 {len(frames)} 帧，前 5 张：", frames[:5]) 输出

import re
import cv2
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from sympy import false

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg"}
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def safe_stem(p: Path) -> str:
    return SAFE_NAME_RE.sub("_", p.stem)


class VideoFrameExtractor:
    def __init__(self, output_dir: str = "frames_out", use_ffmpeg: bool = True):
        self.output_dir = Path(output_dir)
        self.use_ffmpeg = use_ffmpeg and shutil.which("ffmpeg") is not None

    def extract(
        self,
        video_path: str,
        fps: float = 2.0,
        every_n: Optional[int] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        size: Optional[Tuple[int, int]] = None,
        fmt: str = "png",
        jpg_quality: int = 95,
        batch_size: int = 30,
        mode: int = 2,
    ) -> list[str]:

        video = Path(video_path)
        out_dir = self.output_dir / safe_stem(video)
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == 2 or self.use_ffmpeg == False:
            return self._extract_opencv(video, out_dir, fps, every_n, start, end, size, fmt, jpg_quality, batch_size)
        else:
            return self._extract_ffmpeg(video, out_dir, fps, every_n, start, end, size, fmt, jpg_quality)

        #if self.use_ffmpeg:
        #    return self._extract_ffmpeg(video, out_dir, fps, every_n, start, end, size, fmt, jpg_quality)
        #else:
        #    return self._extract_opencv(video, out_dir, fps, every_n, start, end, size, fmt, jpg_quality, batch_size)

    def _extract_ffmpeg(
        self, video: Path, out_dir: Path,
        fps, every_n, start, end, size, fmt, jpg_quality
    ) -> list[str]:
        vf = []
        if size:
            w, h = size
            if w == -1 and h != -1:
                vf.append(f"scale=-2:{h}")
            elif h == -1 and w != -1:
                vf.append(f"scale={w}:-2")
            elif w != -1 and h != -1:
                vf.append(f"scale={w}:{h}")
        if fps:
            vf.append(f"fps={fps}")
        elif every_n:
            vf.append(f"select=not(mod(n\\,{every_n}))")

        pattern = str(out_dir / f"{safe_stem(video)}_%06d.{fmt}")
        cmd = ["ffmpeg", "-y"]
        if start is not None:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(video)]
        if end is not None:
            cmd += ["-to", str(end)]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        if fmt.lower() in ("jpg", "jpeg"):
            cmd += ["-q:v", str(max(2, min(31, 31 - round((jpg_quality/100)*29))))]  # 转换 JPEG 质量
        cmd += [pattern]

        subprocess.run(cmd, check=True)
        return [str(p) for p in sorted(out_dir.glob(f"*.{fmt}"))]

    def _extract_opencv(
        self, video: Path, out_dir: Path,
        fps, every_n, start, end, size, fmt, jpg_quality, batch_size
    ) -> list[str]:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = int(start * native_fps) if start else 0
        end_frame = int(end * native_fps) if end else total

        stride = 1
        if fps and native_fps > 0:
            stride = max(1, int(round(native_fps / fps)))
        elif every_n:
            stride = every_n

        frame_paths = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx_out = 0

        batch_idx = 1
        batch_dir = out_dir / f"batch_{batch_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        while True:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if (pos - start_frame) % stride != 0:
                continue

            if size:
                w, h = size
                if w > 0 and h > 0:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            # 👇 控制 batch 文件夹
            if batch_size and idx_out > 0 and idx_out % batch_size == 0:
                batch_idx += 1
                batch_dir = out_dir / f"batch_{batch_idx}"
                batch_dir.mkdir(parents=True, exist_ok=True)

            out_path = batch_dir / f"{safe_stem(video)}_{idx_out:06d}.{fmt}"
            if fmt.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                cv2.imwrite(str(out_path), frame)

            frame_paths.append(str(out_path))
            idx_out += 1

        cap.release()
        return frame_paths
