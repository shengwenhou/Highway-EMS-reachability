"""Narration audio for the 五楊 animation: one Gemini TTS request per scene of video/narration.md.

Writes video/audio/scene{N}.wav and video/audio/usage.json (tokens per request). A scene whose
WAV already exists is skipped, so editing one paragraph and deleting its WAV re-records only that
scene. Authentication is added by the platform; no key or Authorization header is sent here.
"""
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "audio"
URL = ("https://aiplatform.googleapis.com/v1/projects/claude-cloud-tts/locations/global/"
       "publishers/google/models/gemini-3.8-flash-tts:generateContent")
VOICE = "Charon"
# the style instruction must go in speechMetadata.style; inside the text it would be read aloud
STYLE = "台灣腔的男性旁白，語氣沉穩、清楚，像公共電視的紀錄片，語速中等"


def scenes():
    text = (ROOT / "narration.md").read_text()
    return [p.strip() for p in re.findall(r"\*\*旁白\*\*：\n(.*?)(?=\n## |\Z)", text, re.S)]


def record(n, text):
    body = {"contents": [{"role": "user", "parts": [{"text": text, "speechMetadata": {"style": STYLE}}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"voice": VOICE}}}}
    out = subprocess.run(["curl", "-sS", "-m", "300", "-X", "POST", "-H", "Content-Type: application/json",
                          "--data-binary", "@-", URL],
                         input=json.dumps(body, ensure_ascii=False), capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    if "error" in d:
        sys.exit(f"scene {n}: {d['error'].get('message')}")
    part = d["candidates"][0]["content"]["parts"][0]["inlineData"]
    (AUDIO / f"scene{n}.wav").write_bytes(base64.b64decode(part["data"]))
    return d.get("usageMetadata", {})


def main():
    AUDIO.mkdir(exist_ok=True)
    log_f = AUDIO / "usage.json"
    log = json.loads(log_f.read_text()) if log_f.exists() else {}
    for n, text in enumerate(scenes(), 1):
        if (AUDIO / f"scene{n}.wav").exists():
            continue
        u = record(n, text.replace("\n\n", "\n"))
        log[f"scene{n}"] = {"chars": len(text), "prompt_tokens": u.get("promptTokenCount"),
                            "audio_tokens": u.get("candidatesTokenCount")}
        print(n, log[f"scene{n}"], flush=True)
        log_f.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        time.sleep(3)


if __name__ == "__main__":
    main()
