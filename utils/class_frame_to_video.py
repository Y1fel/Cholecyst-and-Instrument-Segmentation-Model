import cv2
from pathlib import Path
from typing import List, Union

class VideoFrameMerger:
    def __init__(self,
                 frame_dirs: Union[str, List[str]],
                 output_path: str,
                 fps: int = 25,
                 size: tuple = None,
                 fourcc: str = "mp4v",
                 auto_batches: bool = True,
                 log_path: str = "succeed_frames.txt",
                 save_failed_log: bool = True):
        """
        将帧目录合并成视频，并保存日志

        Args:
            frame_dirs (str | list): 帧目录，可以是单个父目录/子目录，或目录列表
            output_path (str): 输出视频路径
            fps (int): 视频帧率
            size (tuple): 输出视频大小 (width, height)，如果 None 就用第一帧大小
            fourcc (str): 编码格式，常用 'mp4v', 'XVID', 'MJPG'
            auto_batches (bool): 是否自动识别父目录下的 batch_x 子目录
            log_path (str): 日志文件路径（.txt），如果 None 就自动生成
            save_failed_log (bool): 是否单独保存失败帧日志
        """
        if isinstance(frame_dirs, str):
            frame_dirs = [frame_dirs]
        self.frame_dirs = [Path(d) for d in frame_dirs]
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.fourcc = fourcc
        self.auto_batches = auto_batches
        self.save_failed_log = save_failed_log

        if log_path is None:
            self.log_path = str(Path(output_path).with_suffix(".txt"))
        else:
            self.log_path = log_path

        if self.save_failed_log:
            self.failed_log_path = str(Path(output_path).with_name("missing_frames.txt"))
        else:
            self.failed_log_path = None

    def _discover_batches(self, parent: Path) -> List[Path]:
        """查找父目录下所有 batch_x 子目录"""
        batch_dirs = [d for d in sorted(parent.iterdir()) if d.is_dir() and d.name.startswith("batch_")]
        return batch_dirs if batch_dirs else [parent]

    def _load_frames(self) -> List[Path]:
        """读取所有帧文件，并按名称排序"""
        frames = []
        for d in self.frame_dirs:
            if self.auto_batches:
                dirs = self._discover_batches(d)
            else:
                dirs = [d]
            for subdir in dirs:
                frames.extend([p for p in subdir.glob("*") if p.suffix.lower() in [".jpg", ".png"]])
        return sorted(frames)

    def merge(self):
        frames = self._load_frames()
        if not frames:
            raise RuntimeError("没有找到任何帧，请检查输入目录！")

        # 获取第一帧大小
        first_frame = cv2.imread(str(frames[0]))
        if self.size is None:
            h, w = first_frame.shape[:2]
            self.size = (w, h)

        # 初始化视频写入器
        fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
        out = cv2.VideoWriter(self.output_path, fourcc_code, self.fps, self.size)

        failed_frames = []

        with open(self.log_path, "w", encoding="utf-8") as log_file:
            for f in frames:
                img = cv2.imread(str(f))
                if img is None:
                    log_file.write(f"无法读取帧: {f}\n")
                    failed_frames.append(str(f))
                    continue
                if (img.shape[1], img.shape[0]) != self.size:
                    img = cv2.resize(img, self.size)
                out.write(img)
                log_file.write(f"{f}\n")

        out.release()

        if self.save_failed_log and failed_frames:
            with open(self.failed_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(failed_frames))

        print(f"视频已保存到: {self.output_path}, 共 {len(frames)} 帧")
        print(f"合并日志已保存到: {self.log_path}")
        if self.save_failed_log and failed_frames:
            print(f"失败帧日志已保存到: {self.failed_log_path}, 共 {len(failed_frames)} 张")
