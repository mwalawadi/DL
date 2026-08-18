import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { PredictionRequest, FormErrors } from '../types/prediction';
import { predictPrice } from '../api/predictionClient';

// ─── Static option lists ─────────────────────────────────────────────────────
const FURNISHING_OPTIONS = ['Furnished', 'Semi-Furnished', 'Unfurnished'];
const TRANSACTION_OPTIONS = ['New Property', 'Resale'];
const OWNERSHIP_OPTIONS = ['Freehold', 'Leasehold', 'Co-operative Society', 'Power Of Attorney'];
const FACING_OPTIONS = ['East', 'West', 'North', 'South', 'North-East', 'North-West', 'South-East', 'South-West'];

const DEFAULT_FORM: PredictionRequest = {
  location: '',
  carpet_area_sqft: '' as unknown as number,
  floor_num: '' as unknown as number,
  bathroom: '' as unknown as number,
  balcony: '' as unknown as number,
  furnishing: '',
  transaction: '',
  ownership: '',
  facing: '',
};

// ─── Validation ───────────────────────────────────────────────────────────────
function validate(form: PredictionRequest): FormErrors {
  const errors: FormErrors = {};
  if (!form.location) errors.location = 'Location is required.';
  if (!form.carpet_area_sqft || form.carpet_area_sqft <= 0)
    errors.carpet_area_sqft = 'Carpet area must be greater than 0.';
  if (form.floor_num === ('' as unknown as number) || form.floor_num < 0)
    errors.floor_num = 'Floor number must be 0 or more.';
  if (!form.bathroom || form.bathroom < 0)
    errors.bathroom = 'At least 0 bathrooms required.';
  if (form.balcony === ('' as unknown as number) || form.balcony < 0)
    errors.balcony = 'Balconies must be 0 or more.';
  if (!form.furnishing) errors.furnishing = 'Please select furnishing status.';
  if (!form.transaction) errors.transaction = 'Please select transaction type.';
  if (!form.ownership) errors.ownership = 'Please select ownership type.';
  if (!form.facing) errors.facing = 'Please select facing direction.';
  return errors;
}

// ─── Component ────────────────────────────────────────────────────────────────
const PredictionForm: React.FC = () => {
  const navigate = useNavigate();
  const [form, setForm] = useState<PredictionRequest>(DEFAULT_FORM);
  const [errors, setErrors] = useState<FormErrors>({});
  const [locations, setLocations] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  // Load location options from locations.json (served as a static asset)
  useEffect(() => {
    fetch('/locations.json')
      .then((r) => r.json())
      .then((data: string[]) => setLocations(data))
      .catch(() => {
        // Fallback: fetch from the backend if the frontend doesn't serve it.
        // Uses VITE_API_BASE_URL — no hardcoded URL.
        const apiBase = import.meta.env.VITE_API_BASE_URL;
        if (!apiBase) {
          setLocations(['Other']);
          return;
        }
        fetch(`${apiBase}/locations.json`)
          .then((r) => r.json())
          .then((data: string[]) => setLocations(data))
          .catch(() => setLocations(['Other']));
      });
  }, []);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value, type } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === 'number' ? (value === '' ? ('' as unknown as number) : Number(value)) : value,
    }));
    // Clear per-field error on change
    if (errors[name as keyof FormErrors]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);

    const validationErrors = validate(form);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }

    setLoading(true);
    try {
      const result = await predictPrice(form);
      // Navigate to result page with prediction data
      navigate('/result', { state: { prediction: result, formData: form } });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Failed to connect to the prediction API.';
      setApiError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="form-grid">

        {/* Location */}
        <div className="form-group full-width">
          <label className="form-label" htmlFor="location">
            Location <span className="required">*</span>
          </label>
          <select
            id="location"
            name="location"
            className={`form-select ${errors.location ? 'error' : ''}`}
            value={form.location}
            onChange={handleChange}
          >
            <option value="">Select location…</option>
            {locations.map((loc) => (
              <option key={loc} value={loc}>{loc}</option>
            ))}
          </select>
          {errors.location && (
            <span className="form-error">⚠ {errors.location}</span>
          )}
        </div>

        {/* Carpet Area */}
        <div className="form-group">
          <label className="form-label" htmlFor="carpet_area_sqft">
            Carpet Area (sq ft) <span className="required">*</span>
          </label>
          <input
            id="carpet_area_sqft"
            name="carpet_area_sqft"
            type="number"
            min="1"
            placeholder="e.g. 1200"
            className={`form-input ${errors.carpet_area_sqft ? 'error' : ''}`}
            value={form.carpet_area_sqft === ('' as unknown as number) ? '' : form.carpet_area_sqft}
            onChange={handleChange}
          />
          {errors.carpet_area_sqft && (
            <span className="form-error">⚠ {errors.carpet_area_sqft}</span>
          )}
        </div>

        {/* Floor */}
        <div className="form-group">
          <label className="form-label" htmlFor="floor_num">
            Floor Number <span className="required">*</span>
          </label>
          <input
            id="floor_num"
            name="floor_num"
            type="number"
            min="0"
            placeholder="0 = Ground Floor"
            className={`form-input ${errors.floor_num ? 'error' : ''}`}
            value={form.floor_num === ('' as unknown as number) ? '' : form.floor_num}
            onChange={handleChange}
          />
          {errors.floor_num && (
            <span className="form-error">⚠ {errors.floor_num}</span>
          )}
        </div>

        {/* Bathrooms */}
        <div className="form-group">
          <label className="form-label" htmlFor="bathroom">
            Bathrooms <span className="required">*</span>
          </label>
          <input
            id="bathroom"
            name="bathroom"
            type="number"
            min="0"
            max="20"
            placeholder="e.g. 2"
            className={`form-input ${errors.bathroom ? 'error' : ''}`}
            value={form.bathroom === ('' as unknown as number) ? '' : form.bathroom}
            onChange={handleChange}
          />
          {errors.bathroom && (
            <span className="form-error">⚠ {errors.bathroom}</span>
          )}
        </div>

        {/* Balconies */}
        <div className="form-group">
          <label className="form-label" htmlFor="balcony">
            Balconies <span className="required">*</span>
          </label>
          <input
            id="balcony"
            name="balcony"
            type="number"
            min="0"
            max="10"
            placeholder="e.g. 1"
            className={`form-input ${errors.balcony ? 'error' : ''}`}
            value={form.balcony === ('' as unknown as number) ? '' : form.balcony}
            onChange={handleChange}
          />
          {errors.balcony && (
            <span className="form-error">⚠ {errors.balcony}</span>
          )}
        </div>

        {/* Furnishing */}
        <div className="form-group">
          <label className="form-label" htmlFor="furnishing">
            Furnishing Status <span className="required">*</span>
          </label>
          <select
            id="furnishing"
            name="furnishing"
            className={`form-select ${errors.furnishing ? 'error' : ''}`}
            value={form.furnishing}
            onChange={handleChange}
          >
            <option value="">Select…</option>
            {FURNISHING_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
          {errors.furnishing && (
            <span className="form-error">⚠ {errors.furnishing}</span>
          )}
        </div>

        {/* Transaction */}
        <div className="form-group">
          <label className="form-label" htmlFor="transaction">
            Transaction Type <span className="required">*</span>
          </label>
          <select
            id="transaction"
            name="transaction"
            className={`form-select ${errors.transaction ? 'error' : ''}`}
            value={form.transaction}
            onChange={handleChange}
          >
            <option value="">Select…</option>
            {TRANSACTION_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
          {errors.transaction && (
            <span className="form-error">⚠ {errors.transaction}</span>
          )}
        </div>

        {/* Ownership */}
        <div className="form-group">
          <label className="form-label" htmlFor="ownership">
            Ownership Type <span className="required">*</span>
          </label>
          <select
            id="ownership"
            name="ownership"
            className={`form-select ${errors.ownership ? 'error' : ''}`}
            value={form.ownership}
            onChange={handleChange}
          >
            <option value="">Select…</option>
            {OWNERSHIP_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
          {errors.ownership && (
            <span className="form-error">⚠ {errors.ownership}</span>
          )}
        </div>

        {/* Facing */}
        <div className="form-group">
          <label className="form-label" htmlFor="facing">
            Facing Direction <span className="required">*</span>
          </label>
          <select
            id="facing"
            name="facing"
            className={`form-select ${errors.facing ? 'error' : ''}`}
            value={form.facing}
            onChange={handleChange}
          >
            <option value="">Select…</option>
            {FACING_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
          {errors.facing && (
            <span className="form-error">⚠ {errors.facing}</span>
          )}
        </div>

      </div>

      {/* API error */}
      {apiError && (
        <div className="error-banner" role="alert">
          <span className="error-banner-icon">🔴</span>
          <div>
            <strong>Connection Error</strong>
            <br />
            {apiError}
            <br />
            <small>Make sure the FastAPI backend is running on port 8000.</small>
          </div>
        </div>
      )}

      <button
        id="submit-prediction"
        type="submit"
        className="btn btn-primary btn-submit"
        disabled={loading}
      >
        {loading ? (
          <>
            <span className="spinner" />
            Predicting…
          </>
        ) : (
          <>
            ✨ Predict House Price
          </>
        )}
      </button>
    </form>
  );
};

export default PredictionForm;
