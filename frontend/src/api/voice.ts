import axios from 'axios';
import { VoiceProcessResponse } from '../types';

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
