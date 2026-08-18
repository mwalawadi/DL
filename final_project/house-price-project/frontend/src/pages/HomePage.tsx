import React from 'react';
import PredictionForm from '../components/PredictionForm';

const HomePage: React.FC = () => {
  return (
    <div className="page-container">
      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">
          <span>🤖</span>
          <span>ML-Powered Valuation</span>
        </div>
        <h1>
          Predict Your{' '}
          <span className="gradient-text">House Price</span>
          <br />
          Instantly
        </h1>
        <p>
          Enter your property details below and our Random Forest model will
          estimate the market price in seconds — trained on{' '}
          <strong>100,000+ real Indian listings</strong>.
        </p>
      </section>

      {/* Form Card */}
      <div className="card">
        <h2 className="form-title">🏠 Property Details</h2>
        <p className="form-subtitle">Fill in all fields marked with * to get an accurate estimate.</p>
        <PredictionForm />
      </div>

      {/* How it works strip */}
      <div className="section-divider">How it works</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
        {[
          { icon: '📝', title: 'Enter Details', desc: 'Fill in location, area, floor, and features.' },
          { icon: '🧠', title: 'AI Analysis', desc: 'Our ML model processes 9 property features.' },
          { icon: '💰', title: 'Get Estimate', desc: 'Receive a price estimate in Indian Rupees.' },
        ].map((step) => (
          <div key={step.title} className="card" style={{ textAlign: 'center', padding: '1.25rem' }}>
            <div style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>{step.icon}</div>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>{step.title}</div>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>{step.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HomePage;
