import re
import cv2
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Callable
import threading, queue

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
        train_fn: Optional[Callable[[list[str]], None]] = None,  # 新增消费者回调函数
    ) -> tuple[list[str], float]:
        """
        train_fn: 一个函数, 负责消费 batch, 例如训练模型:
            def train_fn(batch_frames: list[str]): ...
        """

        video = Path(video_path)
        out_dir = self.output_dir / safe_stem(video)
        out_dir.mkdir(parents=True, exist_ok=True)

        if mode == 2 or self.use_ffmpeg == False:
            frame_paths, native_fps = self._extract_opencv(
                video, out_dir, fps, every_n, start, end, size,
                fmt, jpg_quality, batch_size, train_fn
            )
            return frame_paths, native_fps
        else:
            frame_paths = self._extract_ffmpeg(
                video, out_dir, fps, every_n, start, end, size,
                fmt, jpg_quality
            )
            return frame_paths, fps

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
            cmd += ["-q:v", str(max(2, min(31, 31 - round((jpg_quality / 100) * 29))))]  # JPEG质量控制
        cmd += [pattern]

        subprocess.run(cmd, check=True)
        return [str(p) for p in sorted(out_dir.glob(f"*.{fmt}"))]

    def _extract_opencv(
            self, video: Path, out_dir: Path,
            fps, every_n, start, end, size, fmt, jpg_quality, batch_size,
            train_fn: Optional[Callable[[str], None]] = None  # 回调参数现在是目录路径
    ) -> tuple[list[str], float]:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = 0
        end_frame = 750

        stride = 1
        #if fps and native_fps > 0:
        #    stride = max(1, int(round(native_fps / fps)))
        #elif every_n:
        #    stride = every_n

        frame_paths = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        idx_out = 0

        batch_idx = 1
        batch_dir = out_dir / f"batch_{batch_idx}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_frames = []

        # 消费者线程
        q = queue.Queue()

        def consumer():
            while True:
                batch_dir_path = q.get()
                if batch_dir_path is None:
                    q.task_done()
                    break
                if train_fn:
                    train_fn(batch_dir_path)  # 🔥 回调传目录路径
                q.task_done()

        t = threading.Thread(target=consumer, daemon=True)
        t.start()

        # 生产者
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

            if batch_size and idx_out > 0 and idx_out % batch_size == 0:
                # 当前 batch 满，放入队列，只传目录
                q.put(str(batch_dir))
                batch_frames.clear()

                batch_idx += 1
                batch_dir = out_dir / f"batch_{batch_idx}"
                batch_dir.mkdir(parents=True, exist_ok=True)

            out_path = batch_dir / f"{safe_stem(video)}_{batch_size}_{idx_out:06d}.{fmt}"
            if fmt.lower() in ("jpg", "jpeg"):
                cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            else:
                cv2.imwrite(str(out_path), frame)

            frame_paths.append(str(out_path))
            batch_frames.append(str(out_path))
            idx_out += 1

        cap.release()

        # 最后一个 batch
        if batch_frames:
            q.put(str(batch_dir))  # 🔥 回调最后一个 batch 目录

        q.put(None)  # 结束信号
        q.join()  # 等待消费者线程处理完

        return frame_paths, native_fps

