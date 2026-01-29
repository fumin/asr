import logging
logging.basicConfig()
lg = logging.getLogger()
[lg.removeHandler(h) for h in lg.handlers]
lg.addHandler(logging.StreamHandler())
lg.setLevel(logging.INFO)
lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

import argparse
import queue
import signal
import threading
import time

import apiclient
import funasr
import httplib2
import numpy as np
import oauth2client.service_account
import torch
import zhconv


class Doc:
    def __init__(self, doc_url):
        self.doc_id = self._get_doc_id(doc_url)
        self.service = self._new_service()

        # Do a simple query to make sure everything's all right.
        self.push(" ")

    def push(self, content):
        requests = [{"insertText": {"text": content, "endOfSegmentLocation": {}}}]
        self.service.documents().batchUpdate(documentId=self.doc_id, body={'requests': requests}).execute()

    def _new_service(self):
        cred_path = "coherent-glow-438817-e2-ddcbfaed96fa.json"
        scopes = ["https://www.googleapis.com/auth/documents"]
        cred = oauth2client.service_account.ServiceAccountCredentials.from_json_keyfile_name(cred_path, scopes)
        http_auth = cred.authorize(httplib2.Http())
        return apiclient.discovery.build("docs", "v1", http=http_auth)

    def _get_doc_id(self, doc_url):
        splits = doc_url.split("/")
        return splits[-2]


class Emitter:
    def __init__(self, doc_url):
        self.doc = None
        if doc_url:
            self.doc = Doc(doc_url)

    def push(self, content):
        if content == "":
            return
        if self.doc:
            self.doc.push(content)
        print(content, end="", flush=True)


# This is a hack to implement Queue.shutdown which is only available in Python 3.13.
import secrets
Shutdown = secrets.token_hex(16)


def emit(kill, transcript_queue, emitter):
    dead = False
    while not dead:
        content = transcript_queue.get()
        if content == Shutdown:
            break

        while True:
            dead = is_killed(kill)
            if dead:
                break

            try:
                emitter.push(content)
                break
            except Exception as err:
                logging.error("%s", err)
            time.sleep(1)

    print("")


class Chunk:
    def __init__(self, t, speech, is_final):
        self.t = t  # timestamp in seconds
        self.speech = speech  # numpy array of speech data
        self.is_final = is_final


class Wav:
    def __init__(self, fpath):
        import soundfile
        self.speech, self._sample_rate = soundfile.read(fpath)
        self.i = 0

    def close(self):
        pass

    def get(self, chunk_secs):
        t = self.i / self._sample_rate
        chunk_stride = int(chunk_secs * self._sample_rate)
        end = min(self.i + chunk_stride, self.speech.shape[0])
        speech_chunk = self.speech[self.i : end]
        is_final = (end == self.speech.shape[0])
        self.i = end
        return Chunk(t, speech_chunk, is_final)

    def sample_rate(self):
        return self._sample_rate

    def dtype(self):
        return self.speech.dtype


class Mic:
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate
        self.recorder = self._get_recorder()
        self._dtype = self.recorder.record(1).dtype
        self.start_t = time.time()

    def close(self):
        self.recorder.__exit__(None, None, None)

    def get(self, chunk_secs):
        t = time.time() - self.start_t
        chunk_stride = int(chunk_secs * self._sample_rate)
        d = self.recorder.record(chunk_stride)
        # Simply pick the first channel.
        d = d[:, 0]
        return Chunk(t, d, False)

    def sample_rate(self):
        return self._sample_rate

    def dtype(self):
        return self._dtype

    def _get_recorder(self):
        import soundcard
        mic = soundcard.default_microphone()
        recorder = mic.recorder(samplerate=self._sample_rate)
        recorder.__enter__()
        return recorder


def is_killed(kill):
    try:
        kill.get(block=False)
        return True
    except queue.Empty:
        return False


def produce(kill, audio_queue, src, chunk_secs):
    while True:
        chunk = src.get(chunk_secs)
        if is_killed(kill):
            chunk.is_final = True

        audio_queue.put(chunk)
        if chunk.is_final:
            break


class Punctuator:
    def __init__(self):
        self.model = funasr.AutoModel(model="../punc_ct-transformer_cn-en-common-vocab471067-large", disable_update=True)
        self.raw = ""
        self.comma = 2
        self.fullstop = 3
        self.questionmark = 4
        self.dunhao = 5

    def push(self, new_text):
        self.raw += new_text

        if len(self.raw) < 32:
            return ""
        res = self.model.generate(input=self.raw, disable_pbar=True)
        puncarr = res[0]["punc_array"]
        num_skip_last = 16
        last_punc = -1
        for i in range(len(puncarr)-num_skip_last, 0, -1):
            p = puncarr[i]
            if p == self.fullstop or p == self.comma or p == self.questionmark or p == self.dunhao:
                last_punc = i
                break
        if last_punc == -1:
            return ""

        out = ""
        for i, p in enumerate(puncarr[:last_punc+1]):
            out += self.raw[i]
            if p == self.fullstop:
                out += "。"
            elif p == self.comma:
                out += "，"
            elif p == self.questionmark:
                out += "？"
            elif p == self.dunhao:
                out += "、"
        self.raw = self.raw[last_punc+1:]
        return zhconv.convert(out, "zh-hant")

    def done(self, new_text):
        self.raw += new_text
        if self.raw == "":
            return ""

        res = self.model.generate(input=self.raw, disable_pbar=True)
        punced = res[0]["text"]
        self.raw = ""
        return zhconv.convert(punced, "zh-hant")


class Paraformer:
    def __init__(self):
        self.punctuator = Punctuator()
        self.model = funasr.AutoModel(model="../speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", disable_update=True)
        self.chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
        self.encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
        self.decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

        self.silent_secs = 0

    def decode(self, chunk):
        is_final = chunk.is_final
        if self.silent_secs > 5:
            self.silent_secs = 0
            is_final = True
        res = self.model.generate(
                input=chunk.speech, is_final=is_final, disable_pbar=True,
                chunk_size=self.chunk_size, encoder_chunk_look_back=self.encoder_chunk_look_back, decoder_chunk_look_back=self.decoder_chunk_look_back)
        new_text = res[0]["text"]

        if chunk.is_final or (is_final and new_text == ""):
            punced = self.punctuator.done(new_text)
        else:
            punced = self.punctuator.push(new_text)

        if new_text == "":
            self.silent_secs += self.chunk_secs()
        else:
            self.silent_secs = 0

        return punced

    def chunk_secs(self):
        return self.chunk_size[1] * 0.06


def is_punc(c):
    return "。，、？！：；.,?!:;".find(c) != -1


class Buffer:
    def __init__(self, sample_rate, dtype, window_secs):
        self.sample_rate = sample_rate
        self.window = window_secs * sample_rate
        self.buf = np.zeros(self.window*10, dtype=dtype)
        self.start = 0
        self.end = 0

    def push(self, chunk):
        speech = chunk.speech
        new_end = self.end + speech.shape[0]
        if new_end > self.buf.shape[0]:
            valid = self.end - self.start
            self.buf[0:valid] = self.buf[self.start:self.end]
            self.start = 0
            self.end = valid
            new_end = self.end + speech.shape[0]

        self.buf[self.end : new_end] = speech
        self.end = new_end
        self.start = max(self.end-self.window, 0)

        # Compute timestamp.
        end_t = chunk.t + speech.shape[0]/self.sample_rate
        start_t = end_t - (self.end-self.start)/self.sample_rate

        return self.buf[self.start:self.end], start_t


class Nano:
    def __init__(self, sample_rate, dtype):
        # Edits on .venv/lib/python3.12/site-packages/funasr/models/fun_asr_nano/model.py
        # 1. https://github.com/modelscope/FunASR/pull/2797
        #
        # Edits on .venv/lib/python3.12/site-packages/funasr/auto/auto_model.py
        # 1. https://github.com/FunAudioLLM/Fun-ASR/issues/72
        #    We have to adapt the solution therein and multiply by 16/16000.
        #    This is because vadsegments is in 1/16 frames (slice_padding_audio_samples function in .venv/lib/python3.12/site-packages/funasr/utils/vad_utils.py),
        #    whereas ctc results such as timestamps[i]["start_time"] in below code is in seconds.
        self.m = funasr.AutoModel(
                model="../Fun-ASR-Nano-2512",
                vad_model="../speech_fsmn_vad_zh-cn-16k-common-pytorch",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cpu",
                disable_update=True, disable_pbar=True)

        self._chunk_secs = 5
        self.window_secs = 8
        self.tail_secs = 1
        self.last_maxlen = 3

        self.buf = Buffer(sample_rate, dtype, self.window_secs)
        self.last = []

    def decode(self, chunk):
        audio, audio_t = self.buf.push(chunk)

        # Run inference.
        is_final = chunk.is_final
        if self.m.vad_model is not None:
            is_final = True
        res = self.m.generate([torch.tensor(audio)],
                              # prev_text=self.prev_text,
                              is_final=is_final,
                              disable_pbar=True, batch_size=1)

        # Adjust timestamps to the beginning of the audio stream so that they can be matched against self.last.
        if len(res) == 0 or "timestamps" not in res[0]:
            return ""
        timestamps = res[0]["timestamps"]
        for i, ts in enumerate(timestamps):
            timestamps[i]["start_time"] += audio_t
            timestamps[i]["end_time"] += audio_t
        timestamps = self._timestamps_after_last(timestamps)
        if len(timestamps) == 0:
            return ""
        if chunk.is_final:
            want_str = "".join([x["token"] for x in timestamps])
            return zhconv.convert(want_str, "zh-hant")

        # Cutoff the last few timestamps since they might need future context.
        next_start = audio_t + self.chunk_secs()
        end_cutoff = timestamps[-1]["end_time"] - self.tail_secs
        want_end = len(timestamps)
        for i, ts in enumerate(timestamps):
            # Keep if we won't be seeing it in the next chunk.
            if ts["start_time"] < next_start:
                continue
            elif ts["end_time"] > end_cutoff:
                want_end = i
                break
        timestamps = timestamps[:want_end]

        # Cut suffix at punctuation marks.
        # If not, self.prev_text may end at the middle of some phrase, resulting in serious LLM hallucination.
        want_end = 0
        for i in range(len(timestamps)-1, -1, -1):
            ts = timestamps[i]
            if ts["start_time"] < next_start or is_punc(ts["token"]):
                want_end = i+1
                break
        timestamps = timestamps[:want_end]

        self.last = []
        last_len = min(self.last_maxlen, len(timestamps))
        for i in range(last_len):
            self.last.append(timestamps[-last_len+i])
        want_str = "".join([x["token"] for x in timestamps])
        want_str = zhconv.convert(want_str, "zh-hant")

        return want_str

    def chunk_secs(self):
        return self._chunk_secs

    def _timestamps_after_last(self, timestamps):
        if len(self.last) == 0:
            return timestamps
        # Find token after self.last[-1].
        new_i = -1
        for i, ts in enumerate(timestamps):
            after_end = (ts["start_time"] > self.last[-1]["end_time"])
            start_after = (ts["start_time"] >= self.last[-1]["start_time"])
            different_token = (ts["token"] != self.last[-1]["token"])
            if after_end or (start_after and different_token):
                new_i = i
                break
        if new_i == -1:
            return []

        # Roll new_i forward since the positions of silent tokens such as punctuation are flexible.
        for i in range(len(self.last)):
            lasts = [x["token"] for x in self.last[i:]]
            head = [x["token"] for x in timestamps[new_i:new_i+len(self.last)-i]]

            all_same = True
            for j in range(len(lasts)):
                a, b = lasts[j], head[j]
                if is_punc(a) and is_punc(b):
                    continue
                if a != b:
                    all_same = False
                    break
            if all_same:
                new_i += len(self.last)-i
                break

        timestamps = timestamps[new_i:]
        return timestamps


def consume(kill, transcript_queue, audio_queue, model):
    while True:
        chunk = audio_queue.get()
        if is_killed(kill):
            chunk.is_final = True

        text = model.decode(chunk)
        transcript_queue.put(text)

        if chunk.is_final:
            break

    transcript_queue.put(Shutdown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--document_url", help="google docs url")
    args = parser.parse_args()

    # Set kill.maxsize to the number of threads we are running.
    kill = queue.Queue(3)
    def sigint_handler(sig, frame):
        try:
            for _ in range(kill.maxsize):
                kill.put(True, block=False)
        except queue.Full:
            pass
    signal.signal(signal.SIGINT, sigint_handler)

    emitter = Emitter(args.document_url)
    audio_queue = queue.Queue(128)
    transcript_queue = queue.Queue(8192)

    # src = Wav("testdata/shuiqian.wav")
    src = Mic(16000)

    # model = Paraformer()
    model = Nano(src.sample_rate(), src.dtype())

    threads = []
    threads.append(threading.Thread(target=produce, args=[kill, audio_queue, src, model.chunk_secs()]))
    threads.append(threading.Thread(target=consume, args=[kill, transcript_queue, audio_queue, model]))
    threads.append(threading.Thread(target=emit, args=[kill, transcript_queue, emitter]))
    logging.info("Speech recognition begins")
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    src.close()
    logging.info("exit")


if __name__ == "__main__":
    main()
