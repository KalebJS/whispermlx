## v3.9.2 (2026-03-23)

### Fix

- revert Python 3.14 support — torch 2.8.0 lacks 3.14 wheels

## v3.9.1 (2026-03-23)

### Fix

- include assets in wheel and add Python 3.14 to CI matrix

## v3.9.0 (2026-03-23)

### Feat

- fork whisperx to use mlx
- add progress_callback to transcribe, align, and diarize

### Fix

- remove dead model_bytes read that leaked file handle

## v3.8.2 (2026-03-10)

### Feat

- expose avg_logprob per segment from ctranslate2 beam search

### Fix

- use blank_id parameter instead of hardcoded 0 in trellis and backtrack
- revert #986 wildcard alignment that broke word-level timestamps
- default compute_type to float32 on CPU to avoid float16 ValueError

## v3.8.1 (2026-02-14)

### Feat

- pass --hf_token to WhisperModel for gated model support

### Fix

- propagate --model_dir and --model_cache_only to all model loading paths  (#1285)

## v3.8.0 (2026-02-13)

### Feat

- migrate to pyannote-audio v4 with speaker-diarization-community-1 (#1349)

## v3.7.7 (2026-02-13)

### Fix

- derive SRT/VTT cue times from word-level timestamps (#1347)
- add no_repeat_ngram_size and repetition_penalty options to WhisperModel

## v3.7.6 (2026-01-27)

## v3.7.5 (2026-01-27)

### Feat

- add language-aware sentence tokenization (#1269)
- add hotwords argument to CLI for improved recognition of rare terms

### Fix

- pin huggingface-hub<1.0.0 for pyannote-audio compatibility (#1327)
- add missing comma
- incorrect type annotation in get_writer return value The audio_path attribute that the __call__ method of the ResultWriter class takes is a str, not TextIO

## v3.7.4 (2025-10-16)

## v3.7.3 (2025-10-16)

### Feat

- add Swedish alignment model (#1110)

### Fix

- lock down torch and torchaudio versions (#1265)

## v3.7.2 (2025-10-12)

## v3.7.1 (2025-10-12)

## v3.7.0 (2025-10-10)

### Feat

- add support for python 3.13 (#1256)

## v3.6.0 (2025-10-10)

### Feat

- add centralized logging to replace ad-hoc print statements (#1254)

### Refactor

- rename types.py to schema.py to avoid stdlib conflict

## v3.5.0 (2025-10-08)

### Feat

- update Punkt tokenizer to use pre-trained model and handle missing data

## v3.4.3 (2025-10-01)

### Fix

- restrict pyannote-audio version to avoid compatibility issues (#1242)
- **asr**: load VAD model on correct CUDA device (#835)
- **asr**: load VAD model on correct CUDA device

## v3.4.2 (2025-06-27)

## v3.4.1 (2025-06-25)

### Fix

- speaker embedding bug (#1178)

## v3.4.0 (2025-06-24)

### Feat

- enhance diarization with optional output of speaker embeddings
- add diarize_model arg to CLI  (#1101)

### Refactor

- update type hints in diarization module (PEP 585)

## v3.3.4 (2025-05-03)

### Fix

- remove DiarizationPipeline from public API

### Refactor

- update CLI entry point
- implement lazy loading for module imports in whisperx

## v3.3.3 (2025-05-01)

### Feat

- add version and Python version arguments to CLI
- pass hotwords argument to get_prompt (#1073)
- update build and release workflow to use uv for package installation and publishing
- use uv recommended setup
- update Python compatibility workflow to use uv
- use uv for building package
- add Basque alignment model (#1074)
- add Tagalog (tl - Filipino) Phoneme-based ASR Model (#1067)
- add Latvian align model
- add SegmentData type for temporary processing during alignment

### Fix

- downgrade ctranslate2 dependency version
- update setuptools configuration to include package discovery for whisperx

### Refactor

- update import statements to use explicit module paths across multiple files
- consolidate segment data handling in alignment function
- improve type hints and clean up imports
- remove namespace for consistency

## v3.3.1 (2025-01-08)

### Feat

- include speaker information in  WriteTXT when diarizing

### Fix

- update import statement for conjunctions module

### Refactor

- replace NamedTuple with TranscriptionOptions in FasterWhisperPipeline
- add type hints
- simplify imports for better type inference

## v3.3.0 (2025-01-02)

### Feat

- add build and release workflow
- add Python compatibility testing workflow
- restrict Python versions to 3.9 - 3.12
- use model_dir as cache_dir for wav2vec2 (#681)
- add local_files_only option on whisperx.load_model for offline mode (#867)
- add verbose output (#759)
- update versions for pyannote:3.3.2 and faster-whisper:1.1.0 (#936)
- add support for faster-whisper 1.0.3 (#875)
- update faster-whisper to 1.0.2 (#814)

### Fix

- add UTF-8 encoding when reading README.md
- update README image source and enhance setup.py for long description

## v3.2.0 (2024-12-18)

### Feat

- update Norwegian models (#687)
- add new align models (#922)
- pass model to 3.1 in code
- get rid of pyannote versioning and go to 3.1
- Add merge chunks chunk_size as arguments.

### Fix

- Force ctranslate to version 4.4.0
- update faster-whisper dependencies
- **diarize**: key error on empty track
- ZeroDivisionError when --print_progress True
- correct defaut_asr_options with new options (patch 0.8)
- UnboundLocalError: local variable 'align_language' referenced before assignment
- Bug  in type  hinting

## v3.1.1 (2023-05-13)

## v3.1.0 (2023-05-07)

### Feat

- adding the docker file

## v3.0.2 (2023-05-04)

## v3.0.1 (2023-04-30)

## v3.0.0 (2023-04-25)

## v2.0.1 (2023-04-14)

## v2.0.0 (2023-04-01)

### Fix

- force soundfile version update for mp3 support

## v1.0.0 (2023-02-22)

### Fix

- error when loading huggingface model with embedded language model
