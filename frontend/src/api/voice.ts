import axios from 'axios';
import { VoiceProcessResponse, EmotionInfo } from '../types';

const API_BASE_URL = '/api/v1';

export async function processVoice(
  audioBlob: Blob,
  sessionId?: string,
  returnAudio: boolean = false
): Promise<VoiceProcessResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');

  if (sessionId) {
    formData.append('session_id', sessionId);
  }
  formData.append('return_audio', String(returnAudio));

  const response = await axios.post(`${API_BASE_URL}/voice/process`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function processVoiceWithSession(
  sessionId: string,
  audioBlob: Blob,
  returnAudio: boolean = false
): Promise<VoiceProcessResponse> {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  formData.append('return_audio', String(returnAudio));

  const response = await axios.post(
    `${API_BASE_URL}/voice/process/${sessionId}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
}

// ============ SSE 스트리밍 API ============

export interface StreamEventCallbacks {
  onTranscription?: (data: { text: string; confidence: number }) => void;
  onEmotion?: (data: { emotion: string; korean_label: string; confidence: number }) => void;
  onTextChunk?: (data: { text: string }) => void;
  onAudioChunk?: (data: { audio: string; text: string; format: string }) => void;
  onDone?: (data: { processing_time: number; needs_more_info?: boolean; conversation_phase?: string }) => void;
  onError?: (data: { message: string }) => void;
}

/**
 * 실시간 음성 처리 스트리밍 (SSE)
 *
 * 전화처럼 응답을 문장 단위로 TTS 변환하여 실시간 재생합니다.
 */
export function processVoiceStream(
  audioBlob: Blob,
  sessionId: string | undefined,
  callbacks: StreamEventCallbacks
): { abort: () => void } {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.wav');
  if (sessionId) {
    formData.append('session_id', sessionId);
  }

  const abortController = new AbortController();

  // SSE 연결 (fetch 사용)
  fetch(`${API_BASE_URL}/voice/process/stream`, {
    method: 'POST',
    body: formData,
    signal: abortController.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is null');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE 이벤트 파싱
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 유지

        let currentEvent = '';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);
          } else if (line === '' && currentEvent && currentData) {
            // 이벤트 완료, 콜백 호출
            try {
              const data = JSON.parse(currentData);
              switch (currentEvent) {
                case 'transcription':
                  callbacks.onTranscription?.(data);
                  break;
                case 'emotion':
                  callbacks.onEmotion?.(data);
                  break;
                case 'text_chunk':
                  callbacks.onTextChunk?.(data);
                  break;
                case 'audio_chunk':
                  callbacks.onAudioChunk?.(data);
                  break;
                case 'done':
                  callbacks.onDone?.(data);
                  break;
                case 'error':
                  callbacks.onError?.(data);
                  break;
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
            currentEvent = '';
            currentData = '';
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== 'AbortError') {
        console.error('SSE stream error:', error);
        callbacks.onError?.({ message: error.message });
      }
    });

  return {
    abort: () => abortController.abort(),
  };
}

/**
 * Base64 인코딩된 오디오를 재생하는 오디오 큐 관리자
 * HTMLAudioElement 기반으로 MP3 재생을 안정적으로 지원
 */
export class AudioStreamPlayer {
  private audioQueue: { audio: string; text: string }[] = [];
  private isPlaying = false;
  private currentAudio: HTMLAudioElement | null = null;
  private onPlayingText?: (text: string) => void;

  constructor(onPlayingText?: (text: string) => void) {
    this.onPlayingText = onPlayingText;
  }

  /**
   * 오디오 청크를 큐에 추가
   */
  addAudioChunk(audioBase64: string, text: string): void {
    this.audioQueue.push({ audio: audioBase64, text });
    if (!this.isPlaying) {
      this.playNext();
    }
  }

  /**
   * 다음 오디오 재생
   */
  private playNext(): void {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      this.currentAudio = null;
      return;
    }

    this.isPlaying = true;
    const { audio, text } = this.audioQueue.shift()!;

    try {
      // 재생 중인 텍스트 알림
      this.onPlayingText?.(text);

      // HTMLAudioElement를 사용하여 MP3 재생
      const audioElement = new Audio(`data:audio/mp3;base64,${audio}`);
      this.currentAudio = audioElement;

      audioElement.onended = () => {
        this.playNext();
      };

      audioElement.onerror = (e) => {
        console.error('Audio playback error:', e);
        // 에러 발생 시 다음 오디오로 진행
        this.playNext();
      };

      audioElement.play().catch((error) => {
        console.error('Audio play failed:', error);
        this.playNext();
      });
    } catch (error) {
      console.error('Audio playback error:', error);
      // 에러 발생 시 다음 오디오로 진행
      this.playNext();
    }
  }

  /**
   * 재생 중지 및 큐 초기화
   */
  stop(): void {
    this.audioQueue = [];
    this.isPlaying = false;
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.src = '';
      this.currentAudio = null;
    }
  }

  /**
   * 큐가 비어있고 재생이 끝났는지 확인
   */
  isFinished(): boolean {
    return !this.isPlaying && this.audioQueue.length === 0;
  }
}
