import React, { useState } from 'react';

export default function AuthScreen({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const endpoint = isLogin ? '/auth/login' : '/auth/register';
    
    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
        credentials: 'include' // crucial for HttpOnly cookie
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }
      
      if (isLogin) {
        onLogin();
      } else {
        // Automatically switch to login on success
        setIsLogin(true);
        setError('Registration successful. Please login.');
        setTimeout(() => setError(''), 3000);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-header">
        <h1>{isLogin ? 'Access Portal' : 'Initialize User'}</h1>
        <p>{isLogin ? 'Authenticate to AstroLab AI' : 'Create an AstroLab AI profile'}</p>
      </div>
      
      {error && <div className="auth-error">{error}</div>}
      
      <form className="auth-form" onSubmit={handleSubmit}>
        <div className="input-group">
          <label>IDENTIFIER</label>
          <input 
            type="text" 
            className="auth-input" 
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoComplete="off"
            spellCheck="false"
          />
        </div>
        
        <div className="input-group">
          <label>ACCESS KEY</label>
          <input 
            type="password" 
            className="auth-input"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        
        <button type="submit" className="auth-button">
          {isLogin ? 'AUTHENTICATE' : 'REGISTER'}
        </button>
      </form>
      
      <div className="auth-switch">
        {isLogin ? "Don't have an account? " : "Already initialized? "}
        <span onClick={() => { setIsLogin(!isLogin); setError(''); }}>
          {isLogin ? 'Register' : 'Login'}
        </span>
      </div>
    </div>
  );
}
