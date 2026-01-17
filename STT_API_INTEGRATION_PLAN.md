# STT API Integration Plan

## Goal
- Replace or add an STT provider that calls the external ASR API:
  - `POST /api/v1/asr?include_emotion=true`
  - Multipart form field: `audio=@file.wav`
- Optionally use the API-provided emotion result in the voice pipeline.

## Proposed Changes
1) New STT implementation
   - Add `app/services/stt/sensevoice.py` (name can be adjusted).
   - Use an HTTP client to:
     - POST audio bytes as multipart (`audio`).
     - Include `include_emotion` query param (default true).
     - Parse JSON response into `TranscriptionResult`.
   - Add timeouts and clear error messages for network failures or bad responses.

2) Configuration updates
   - Extend `Settings` with:
     - `stt_provider` literal includes a new value (e.g., `sensevoice`).
     - `ASR_API_BASE_URL` (e.g., `http://34.130.232.40:8000`).
     - `ASR_INCLUDE_EMOTION` (bool) and `ASR_TIMEOUT_SECONDS` (float).
   - Document env vars in `.env` or `README.md` (optional but recommended).

3) STT factory wiring
   - Register the new provider in `app/services/stt/factory.py`.
   - Export the class in `app/services/stt/__init__.py`.

4) TranscriptionResult payload (optional)
   - If the API returns emotion data, add an optional field like
     `emotion: Optional[str]` or `extra: dict` to `TranscriptionResult`
     so the data can flow upward without breaking existing code.

5) Pipeline emotion handling (optional)
   - In `app/pipelines/voice_rag_pipeline.py`, if the transcription
     contains emotion and a config flag is enabled, skip the local
     `AudioEmotionAnalyzer` and build an `EmotionResult` from the API
     response instead.

6) Dependencies
   - If using `requests`, add it to `requirements.txt`.
   - Alternative: use `httpx` (also add to requirements).

## Assumptions / Questions
- Confirm the exact JSON response shape from the ASR API:
  - Required keys for text (e.g., `text`, `transcription`, or `result`).
  - Emotion payload structure (label, scores, confidence).
- Confirm whether the API expects WAV only, and any size/time limits.

## Rollout Steps
1) Implement the new STT class and wire it in the factory.
2) Add config entries and update env examples.
3) (Optional) Pipeline emotion shortcut using API emotion.
4) Quick manual test with a local WAV file.
