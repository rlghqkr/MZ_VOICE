import apiClient from './client';
import { Session, SessionEndResult } from '../types';

export async function createSession(customerEmail?: string): Promise<Session> {
  const response = await apiClient.post('/sessions', {
    customer_email: customerEmail,
  });
  return response.data;
}

export async function getSession(sessionId: string): Promise<Session> {
  const response = await apiClient.get(`/sessions/${sessionId}`);
  return response.data;
}

export async function updateSessionEmail(sessionId: string, email: string): Promise<void> {
  await apiClient.put(`/sessions/${sessionId}/email`, { email });
}

export async function endSession(sessionId: string): Promise<SessionEndResult> {
  const response = await apiClient.post(`/sessions/${sessionId}/end`);
  return response.data;
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiClient.delete(`/sessions/${sessionId}`);
}
