import React, { useState, useEffect } from 'react';
import AuthScreen from './components/AuthScreen';
import TerminalScreen from './components/TerminalScreen';
import { API_BASE_URL } from './config';
import './index.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated by calling /me
    fetch(`${API_BASE_URL}/auth/me`, {credentials: 'include'})
      .then(res => {
        if (res.ok) {
          setIsAuthenticated(true);
        }
      })
      .catch(err => console.error("Auth check failed", err))
      .finally(() => setLoading(false));
  }, []);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    fetch(`${API_BASE_URL}/auth/logout`, { method: 'POST', credentials: 'include' })
      .then(() => setIsAuthenticated(false));
  };

  if (loading) return null; // Avoid flicker

  return (
    <>
      {isAuthenticated ? (
        <TerminalScreen onLogout={handleLogout} />
      ) : (
        <AuthScreen onLogin={handleLogin} />
      )}
    </>
  );
}

export default App;
