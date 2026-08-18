/**
 * API client for the House Price Prediction backend.
 *
 * The base URL is read from the VITE_API_BASE_URL environment variable.
 * Copy frontend/.env.example → frontend/.env and set VITE_API_BASE_URL
 * before running `npm run dev`.
 */

import axios from 'axios';
import type { PredictionRequest, PredictionResponse } from '../types/prediction';

const BASE_URL: string = import.meta.env.VITE_API_BASE_URL;

if (!BASE_URL) {
  throw new Error(
    '[predictionClient] VITE_API_BASE_URL is not set. ' +
    'Copy frontend/.env.example to frontend/.env and set the variable.'
  );
}

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
