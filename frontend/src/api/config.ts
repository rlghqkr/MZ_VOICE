import apiClient from './client';

export interface AppConfig {
  stt_provider: string;
  tts_provider: string;
  llm_model: string;
  embedding_model: string;
}

export interface PipelineStats {
  stt_provider: string;
  tts_provider: string;
  use_langgraph: boolean;
  rag_stats: {
    collection_name: string;
    document_count: number;
    llm_model: string;
    embedding_model: string;
  } | null;
}

export async function getConfig(): Promise<AppConfig> {
  const response = await apiClient.get('/config');
  return response.data;
}

export async function getStats(): Promise<PipelineStats> {
  const response = await apiClient.get('/stats');
  return response.data;
}

export async function healthCheck(): Promise<{ status: string; version: string }> {
  const response = await apiClient.get('/health');
  return response.data;
}
