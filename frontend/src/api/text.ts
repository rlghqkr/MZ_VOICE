import apiClient from './client';
import { TextProcessResponse, EmotionType } from '../types';

export async function processText(
  text: string,
  emotion: EmotionType = 'neutral',
  sessionId?: string,
  returnAudio: boolean = false
): Promise<TextProcessResponse> {
  const response = await apiClient.post('/text/process', {
    text,
    emotion,
    session_id: sessionId,
    return_audio: returnAudio,
  });

  return response.data;
}

export async function processTextWithSession(
  sessionId: string,
  text: string,
  emotion: EmotionType = 'neutral',
  returnAudio: boolean = false
): Promise<TextProcessResponse> {
  const response = await apiClient.post(`/text/process/${sessionId}`, {
    text,
    emotion,
    return_audio: returnAudio,
  });

  return response.data;
}
