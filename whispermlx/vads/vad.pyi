class Vad:
    def __init__(self, vad_onset: float) -> None: ...
    @staticmethod
    def preprocess_audio(audio: object) -> object: ...
    @staticmethod
    def merge_chunks(
        segments: list,
        chunk_size: int,
        onset: float,
        offset: float | None,
    ) -> list[dict]: ...
