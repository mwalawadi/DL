import React from 'react';
import { useLocation, Link, Navigate } from 'react-router-dom';
import type { PredictionResponse, PredictionRequest } from '../types/prediction';

interface LocationState {
  prediction: PredictionResponse;
  formData: PredictionRequest;
}

/**
 * Format a price in INR into a human-friendly string.
 * Example: 4_250_000 → "₹ 42.5 Lac"
 */
function formatINR(price: number): { main: string; sub: string } {
  if (price >= 1e7) {
    return {
      main: `₹ ${(price / 1e7).toFixed(2)} Cr`,
      sub: `₹ ${price.toLocaleString('en-IN')}`,
    };
  }
  return {
    main: `₹ ${(price / 1e5).toFixed(2)} Lac`,
    sub: `₹ ${price.toLocaleString('en-IN')}`,
  };
}

const ResultPage: React.FC = () => {
  const location = useLocation();
  const state = location.state as LocationState | null;

  // Guard: redirect home if there's no prediction data
  if (!state?.prediction) {
    return <Navigate to="/" replace />;
  }

  const { prediction, formData } = state;
  const { main, sub } = formatINR(prediction.predicted_price);

  return (
    <div className="page-container">
      <div className="result-page">
        <div className="result-icon">🏡</div>

        <div className="result-card">
          <p className="result-label">Estimated Market Value</p>
          <div className="result-price">{main}</div>
          <p className="result-price-sub">{sub}</p>

          <div className="result-divider" />

          {/* Property summary */}
          <div style={{ textAlign: 'left', fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
              {[
                ['📍 Location', formData.location],
                ['📐 Area', `${formData.carpet_area_sqft} sq ft`],
                ['🏢 Floor', formData.floor_num === 0 ? 'Ground' : `Floor ${formData.floor_num}`],
                ['🚿 Bathrooms', formData.bathroom],
                ['🏗️ Furnishing', formData.furnishing],
                ['🧭 Facing', formData.facing],
              ].map(([label, value]) => (
                <div key={String(label)}>
                  <span style={{ color: 'var(--color-text-muted)' }}>{label}</span>
                  <br />
                  <strong style={{ color: 'var(--color-text-primary)' }}>{value}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="result-divider" />

          <p className="result-note">
            ⚠️ This is an AI-generated estimate based on historical listing data.
            Actual transaction prices may vary depending on market conditions and
            property inspection.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '2rem', flexWrap: 'wrap', justifyContent: 'center' }}>
          <Link to="/" className="btn btn-primary" id="predict-again-btn">
            🔄 Predict Another
          </Link>
          <button
            className="btn btn-secondary"
            id="copy-result-btn"
            onClick={() => {
              navigator.clipboard?.writeText(`Estimated house price: ${main} (${sub})`);
            }}
          >
            📋 Copy Result
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultPage;
