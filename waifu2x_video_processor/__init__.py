#!/bin/python3

import argparse
import dataclasses
import os
import pathlib
import psutil
import queue
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import threading

from typing import List, Sequence, Tuple

DEFAULT_SCALE_RATIO = 2.0
DEFAULT_NOISE_LEVEL = None
DEFAULT_VIDEO_STREAM = 0
DEFAULT_MEDIAINFO_PATH = pathlib.Path("mediainfo")
DEFAULT_FFMPEG_PATH = pathlib.Path("ffmpeg")
DEFAULT_WAIFU2X_PATH = pathlib.Path("waifu2x-converter-cpp")
DEFAULT_PROCESSORS = (0,)


@dataclasses.dataclass
class Waifu2xJob:
    input_path: pathlib.Path
    output_path: pathlib.Path


@dataclasses.dataclass
class FrameExtractJob:
    output_dir: pathlib.Path
    start_frame_no: int
    frame_count: int


@dataclasses.dataclass
class State:
    frame_count: int = 0
    frame_extraction_index: int = 0
    frame_output_index: int = 0
    processors: Tuple[int, ...] = DEFAULT_PROCESSORS
    processor_scores: List[float] = dataclasses.field(
        default_factory=lambda: [1.0 / len(DEFAULT_PROCESSORS)] * len(DEFAULT_PROCESSORS)
    )

    def print_state(self, file=sys.stderr):
        print(
            f"Frames extracted: {self.frame_extraction_index}/{self.frame_count}\n"
            f"Frames processed: {self.frame_output_index}/{self.frame_count}\n"
            f"Processor scores:",
            file=file
        )
        for i, proc in enumerate(self.processors):
            print(f"    Processor {proc}: {self.processor_scores[i]}", file=file)


_state = State()


class MovingAverage:
    _data: List[float]
    _window_size: int

    def __init__(self, window_size: int):
        self._data = []
        self._window_size = window_size

    def push(self, value: float):
        self._data.append(value)
        if len(self._data) > self._window_size:
            del self._data[0]

    @property
    def mean(self) -> float:
        return sum(self._data) / len(self._data)


def w2x_runner(
        w2x_path: pathlib.Path,
        job: Waifu2xJob, processor: int = None,
        scale_ratio: float = None, noise_level: int = None,
):
    cmd = [
        w2x_path,
        "--png-compression", "3",
        "--processor", str(processor),
        "--input", job.input_path.as_posix(),
        "--output", job.output_path.as_posix()
    ]

    if (scale_ratio is None) and (noise_level is None):
        shutil.copyfile(job.input_path, job.output_path)

    elif (scale_ratio is not None) and (noise_level is not None):
        cmd += [
            "--mode", "noise-scale",
            "--scale-ratio", str(scale_ratio),
            "--noise-level", str(noise_level)
        ]

    elif scale_ratio is not None:
        cmd += [
            "--mode", "scale",
            "--scale-ratio", str(scale_ratio),
        ]

    elif noise_level is not None:
        cmd += [
            "--mode", "noise",
            "--noise-level", str(noise_level)
        ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()


def w2x_thread(
        w2x_path: pathlib.Path,
        jobs: queue.Queue, times: queue.Queue,
        processor: int = None, scale_ratio: float = None, noise_level: int = None
):
    while 1:
        job = jobs.get()
        start = time.perf_counter()
        w2x_runner(w2x_path, job, processor, scale_ratio, noise_level)
        times.put(time.perf_counter() - start)
        jobs.task_done()


def frame_extractor(
        ffmpeg_path: pathlib.Path,
        output_dir: pathlib.Path, video_in_path: pathlib.Path, video_stream: int
) -> subprocess.Popen:
    cmd = [
        ffmpeg_path,
        "-i", video_in_path.as_posix(),
        "-map", f"0:{video_stream}",
        "-c:v", "libwebp",
        "-lossless", "1",
        f"{output_dir.as_posix()}/%d.webp"
    ]

    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_frame_count(video_in_path: pathlib.Path, mediainfo_path: pathlib.Path) -> int:
    cmd = [mediainfo_path, r'--Output=Video;%FrameCount%', video_in_path.resolve().as_posix()]
    return int(subprocess.run(cmd, capture_output=True).stdout)


def benchmark_processor(
        ffmpeg_path: pathlib.Path, waifu2x_path: pathlib.Path,
        video_in_path: pathlib.Path, video_stream: int,
        processor: int, scale_ratio: float = None, noise_level: int = None,

) -> float:
    with tempfile.TemporaryDirectory() as _td:
        td = pathlib.Path(_td)
        cmd = [
            ffmpeg_path,
            "-i", video_in_path,
            "-map", f"0:{video_stream}",
            "-c:v", "libwebp",
            "-lossless", "1",
            "-vframes", "1",
            f"{td}/orig%d.webp"
        ]
        subprocess.run(cmd, capture_output=True)

        start = time.perf_counter()
        w2x_runner(waifu2x_path, Waifu2xJob(td / "orig1.webp", td / "out1.png"), processor, scale_ratio, noise_level)
        elapsed = time.perf_counter() - start

    print(elapsed)
    return elapsed


def calculate_processor_scores(processor_times: List[MovingAverage]) -> List[float]:
    if len(processor_times) == 1:
        return [1.0]

    times = []
    for t in processor_times:
        times.append(t.mean)
    total_time = sum(times)
    for i in range(len(times)):
        _state.processor_scores[i] = total_time / times[i]
    return _state.processor_scores


def main(
        video_in_path: pathlib.Path,
        scale_ratio: float = DEFAULT_SCALE_RATIO,
        noise_level: int = DEFAULT_NOISE_LEVEL,
        processors: Sequence[int] = DEFAULT_PROCESSORS,
        video_stream: int = DEFAULT_VIDEO_STREAM,
        ffmpeg_path: pathlib.Path = DEFAULT_FFMPEG_PATH,
        mediainfo_path: pathlib.Path = DEFAULT_MEDIAINFO_PATH,
        waifu2x_path: pathlib.Path = DEFAULT_WAIFU2X_PATH
) -> int:
    _state.frame_count = get_frame_count(video_in_path, mediainfo_path)
    _state.processors = tuple(processors)
    # _state.processors = tuple([p for sublist in zip(processors, processors) for p in sublist])
    _state.processor_scores = [1.0 / len(processors)] * len(processors)
    _state.frame_output_index = 0
    _state.frame_extraction_index = 0
    frame_buffer_limit = 50
    ffmpeg_paused = False

    processor_times = []

    for processor in processors:
        processor_times.append(MovingAverage(10))
        processor_times[-1].push(
            benchmark_processor(ffmpeg_path, waifu2x_path, video_in_path, video_stream, processor, scale_ratio,
                                noise_level)
        )

    calculate_processor_scores(processor_times)
    _state.print_state()
    print("Send SIGUSR1 to get an update on this information.", file=sys.stderr)

    w2x_threads = []
    w2x_queues = []
    w2x_times = []

    for i in range(len(processors)):
        w2x_queues.append(queue.Queue())
        w2x_times.append(queue.Queue())
        w2x_threads.append(
            threading.Thread(
                target=w2x_thread,
                args=(waifu2x_path, w2x_queues[-1], w2x_times[-1], processors[i], scale_ratio, noise_level),
                daemon=True
            )
        )
        w2x_threads[-1].start()

    with tempfile.TemporaryDirectory() as _td:
        td = pathlib.Path(_td)
        original_dir = td / "original"
        preprocess_dir = td / "preprocess"
        processed_dir = td / "processed"

        original_dir.mkdir()
        preprocess_dir.mkdir()
        processed_dir.mkdir()

        ffmpeg = frame_extractor(ffmpeg_path, original_dir, video_in_path, video_stream)
        ffmpeg_proc = psutil.Process(ffmpeg.pid)

        while _state.frame_output_index < _state.frame_count:
            original_count = len(os.listdir(original_dir)) + len(os.listdir(preprocess_dir))
            if ffmpeg.poll() is None:
                if original_count >= frame_buffer_limit:
                    if not ffmpeg_paused:
                        ffmpeg.send_signal(signal.SIGSTOP)
                        ffmpeg_paused = True
                elif ffmpeg_paused:
                    ffmpeg.send_signal(signal.SIGCONT)
                    ffmpeg_paused = False

            calculate_processor_scores(processor_times)

            for i in range(_state.frame_extraction_index, _state.frame_count):
                original = original_dir / f"{i + 1}.webp"

                if not original.is_file():
                    break

                preprocess_path = preprocess_dir / f"{i + 1}.webp"
                processed = processed_dir / f"{i + 1}.png"

                if ffmpeg.poll() is not None:
                    pass
                else:
                    is_open = False
                    for file in ffmpeg_proc.open_files():
                        if original == file.path:
                            # The file is still open by ffmpeg, so it's not done being extracted yet.
                            is_open = True
                            break
                    if is_open:
                        break

                original.rename(preprocess_path)
                _state.frame_extraction_index += 1

                proc_choice = processors.index(random.choices(processors, _state.processor_scores, k=1)[0])
                w2x_queues[proc_choice].put(Waifu2xJob(preprocess_path, processed))

            for i in range(_state.frame_output_index, _state.frame_count):
                processed_path = processed_dir / f"{i + 1}.png"

                if not processed_path.is_file():
                    break

                mod_time = processed_path.stat().st_mtime
                # A crude method of checking if waifu2x is done writing the file.
                if (time.time() - mod_time) < 3:
                    break

                with open(processed_path, "rb") as f:
                    d = f.read()
                    # with open(processed_path.with_suffix(".out.png"), "wb") as f2:
                    #     f2.write(d)
                    sys.stdout.buffer.write(d)
                    # print(f"{processed_path}")

                (preprocess_dir / f"{i + 1}.webp").unlink()
                processed_path.unlink()
                _state.frame_output_index += 1

            for i, times in enumerate(w2x_times):
                try:
                    while 1:
                        processor_times[i].push(times.get(False))
                except queue.Empty:
                    continue

            time.sleep(0.1)
    return 0


def _entry():
    signal.signal(signal.SIGUSR1, lambda s, f: _state.print_state())

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Processes video frames through waifu2x, and outputs them to stdout.",
        epilog=f"To combine this with ffmpeg to produce an actual video output, "
               f"use something like the following, "
               f"noting that it is important to specify the image2pipe input format, "
               f"the framerate, and a stdin (-) input:\n\n"
               f"python3 {sys.argv[0]} my_video.m2ts | ffmpeg -f image2pipe -framerate 23.976 -i - -c:v libx264 "
               f"-crf 15 -vf \"format=yuv420p\" my_bigger_video.mkv\n\n"
    )

    p.add_argument(
        "--scale-ratio",
        type=float,
        help=f"Ratio by which to scale the video. Default is {DEFAULT_SCALE_RATIO}.",
        default=DEFAULT_SCALE_RATIO
    )

    p.add_argument(
        "--noise-level",
        type=int,
        help=f"Noise reduction level. Default is {DEFAULT_NOISE_LEVEL}.",
        default=DEFAULT_NOISE_LEVEL
    )

    p.add_argument(
        "--processors",
        type=lambda s: tuple(map(int, s.split(","))),
        help="waifu2x processors to use. "
             "See `waifu2x-converter-cpp --list-processor`. "
             "Supplied as a comma-separated list of integers, e.g. `0,1` for processors 0 and 1. "
             f"Default is {' '.join([str(p) for p in DEFAULT_PROCESSORS])}.",
        default=DEFAULT_PROCESSORS
    )

    p.add_argument(
        "--video-stream",
        type=int,
        help=f"Stream number to use as the video source in ffmpeg. Default is {DEFAULT_VIDEO_STREAM}.",
        default=DEFAULT_VIDEO_STREAM
    )

    p.add_argument(
        "--mediainfo-path",
        help=f"Path to mediainfo binary. Default is {DEFAULT_MEDIAINFO_PATH}.",
        default=DEFAULT_MEDIAINFO_PATH,
        type=pathlib.Path
    )

    p.add_argument(
        "--ffmpeg-path",
        help=F"Path to ffmpeg binary. Default is {DEFAULT_FFMPEG_PATH}.",
        default=DEFAULT_FFMPEG_PATH,
        type=pathlib.Path
    )

    p.add_argument(
        "--waifu2x-path",
        help=F"Path to waifu2x binary. Default is {DEFAULT_WAIFU2X_PATH}.",
        default=DEFAULT_WAIFU2X_PATH,
        type=pathlib.Path
    )

    p.add_argument(
        "input_path",
        help="Path of the video to process.",
        type=pathlib.Path
    )

    parsed = vars(p.parse_args())
    video = parsed.pop("input_path")

    try:
        main(video, **parsed)
    except KeyboardInterrupt:
        sys.exit(2)
    except BrokenPipeError:
        print("Broken pipe. FFmpeg probably failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _entry()
