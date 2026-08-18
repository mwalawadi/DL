/** TypeScript types for the prediction API. */

/** Request body sent to POST /predict */
export interface PredictionRequest {
  location: string;
  carpet_area_sqft: number;
  floor_num: number;
  bathroom: number;
  balcony: number;
  furnishing: string;
  transaction: string;
  ownership: string;
  facing: string;
}

/** Successful response from POST /predict */
export interface PredictionResponse {
  predicted_price: number;
}

/** Validation errors stored per form field */
export type FormErrors = Partial<Record<keyof PredictionRequest, string>>;
