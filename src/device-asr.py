import torch
import torchaudio


# The data acquisition process will stop after this number of steps.
# This eliminates the need of process synchronization and makes this
# tutorial simple.
NUM_ITER = 100


def stream(q, format, src, segment_length, sample_rate):
    from torchaudio.io import StreamReader

    print("Building StreamReader...")
    streamer = StreamReader(src, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

    print(streamer.get_src_stream_info(0))
    print(streamer.get_out_stream_info(0))

    print("Streaming...")
    print()
    stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
    for _ in range(NUM_ITER):
        (chunk,) = next(stream_iterator)
        q.put(chunk)


class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width

        self.state = None
        self.hypothesis = None

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        hypos, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypothesis
        )
        self.hypothesis = hypos[0]
        transcript = self.token_processor(self.hypothesis[0], lstrip=False)
        return transcript


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context


def main(device, src, bundle):
    print(torch.__version__)
    print(torchaudio.__version__)

    print("Building pipeline...")
    pipeline = Pipeline(bundle)

    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length

    print(f"Sample rate: {sample_rate}")
    print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
    print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

    cacher = ContextCacher(segment_length, context_length)

    @torch.inference_mode()
    def infer():
        for _ in range(NUM_ITER):
            chunk = q.get()
            segment = cacher(chunk[:, 0])
            transcript = pipeline.infer(segment)
            print(transcript, end="", flush=True)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, device, src, segment_length, sample_rate))
    p.start()
    infer()
    p.join()


if __name__ == "__main__":
    main(
        device="dshow",
        src="audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{E97F871E-F5BA-4666-B9F2-B3327CC3322E}",
        bundle=torchaudio.pipelines.Wav2Vec2ASRBundle,
    )

