import React from 'react';
import { Link } from 'react-router-dom';

const NotFoundPage: React.FC = () => {
  return (
    <div className="page-container">
      <div className="not-found">
        <div className="not-found-code">404</div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.5rem' }}>Page Not Found</h2>
        <p style={{ color: 'var(--color-text-secondary)', maxWidth: '320px', textAlign: 'center' }}>
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link to="/" className="btn btn-primary" id="go-home-btn" style={{ marginTop: '1rem' }}>
          🏠 Back to Home
        </Link>
      </div>
    </div>
  );
};

export default NotFoundPage;
