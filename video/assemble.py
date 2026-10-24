"""Mix the narration onto the timeline and encode the final film.

usage: python3 video/assemble.py <frames dir> <out.mp4>
Each video/audio/scene{N}.wav starts at its scene's audio_at (video/data/film.json); the mix is
loudness-normalised (about -14 LUFS once on both channels). Frames come from render_film.js at 30 fps.
"""
import json
import subprocess
import sys
from pathlib import Path

VID = Path(__file__).resolve().parent
FPS = 30


def main():
    frames, out = sys.argv[1], sys.argv[2]
    d = json.load(open(VID / "data" / "film.json"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", f"{frames}/f%05d.jpg"]
    for s in d["scenes"]:
        cmd += ["-i", str(VID / "audio" / f"scene{s['n']}.wav")]
    delays = "".join(f"[{i + 1}:a]adelay={int(s['audio_at'] * 1000)}:all=1[a{i}];" for i, s in enumerate(d["scenes"]))
    mix = "".join(f"[a{i}]" for i in range(len(d["scenes"])))
    filt = (delays + f"{mix}amix=inputs={len(d['scenes'])}:normalize=0,"
            f"apad,atrim=0:{d['total']},loudnorm=I=-17:TP=-1.5:LRA=11,aresample=48000,"
            # the same voice on both channels: some players stay silent on a mono AAC track
            f"pan=stereo|c0=c0|c1=c0[aout]")
    cmd += ["-filter_complex", filt, "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
    subprocess.run(cmd, check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
