/**
 * API client for the House Price Prediction backend.
 *
 * The base URL is read from the VITE_API_BASE_URL environment variable so
 * that no API URL is ever hard-coded inside the React components.
 */

import axios from 'axios';
import type { PredictionRequest, PredictionResponse } from '../types/prediction';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15_000,
});

/**
 * POST /predict — send house features and receive predicted price.
 *
 * @param data  Validated prediction request payload.
 * @returns     Prediction response containing `predicted_price` in INR.
 */
export async function predictPrice(data: PredictionRequest): Promise<PredictionResponse> {
  const response = await api.post<PredictionResponse>('/predict', data);
  return response.data;
}

/**
 * GET /health — verify the backend is reachable.
 *
 * @returns True when the API responds with status "ok".
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get<{ status: string }>('/health');
    return response.data.status === 'ok';
  } catch {
    return false;
  }
}
