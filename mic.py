import logging
logging.basicConfig()
lg = logging.getLogger()
[lg.removeHandler(h) for h in lg.handlers]
lg.addHandler(logging.StreamHandler())
lg.setLevel(logging.INFO)
lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

import numpy as np
import soundcard
import soundfile


def main():
    channels = 1
    samplerate = 16000
    subtype = "PCM_16"
    numframes = int(0.6 * samplerate)

    data = []
    mic = soundcard.default_microphone()
    with mic.recorder(samplerate=samplerate) as recorder:
        for i in range(10):
            d = recorder.record(numframes)
            data.append(d)

    with soundfile.SoundFile("qq.wav", mode="w", channels=channels, samplerate=samplerate, subtype=subtype) as f:
        for d in data:
            f.write(d)


if __name__ == "__main__":
    main()
