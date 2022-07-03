# waifu2x-video-processor

Processes video frames through waifu2x, and outputs them to stdout without needing to initially extract all the frames from the source video.
This can save hundreds of GB for long videos.


## Installation

Make sure you have [waifu2x-converter-cpp](https://github.com/DeadSix27/waifu2x-converter-cpp) installed, or another one with the same CLI.

Clone this repository, then run `pip install ./waifu2x-video-processor`.


## Usage

`waifu2x-video-processor --help`

Typical example:

`waifu2x-video-processor 1080p.m2ts | ffmpeg -f image2pipe -framerate 23.976 -i - -c:v libx264 -crf 15 -vf "format=yuv420p" 2160p.mkv`

`-f image2pipe` is required, make sure `-framerate` is set to the source video's frame rate, and specify stdin (`-`) as the first input.
