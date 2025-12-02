// ===========================================
// LOVABLE FRONTEND INTEGRATION CODE
// Copy and use this code in your Lovable app
// ===========================================

// Your Backend URL - UPDATE THIS if it changes
const API_BASE_URL = 'https://2d09f74f-9771-4c85-b6d6-a965d884c276-00-3q1c8s2ku1arv.kirk.replit.dev';

// ===========================================
// API SERVICE FUNCTIONS
// ===========================================

// Check if backend is running
export async function checkBackendStatus() {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Connect a new user account
export async function connectAccount(userData) {
  try {
    const response = await fetch(`${API_BASE_URL}/connect_account`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: userData.username,
        mt5_login: userData.mt5_login || null,
        mt5_password: userData.mt5_password || null,
        mt5_server: userData.mt5_server || null,
        binance_api_key: userData.binance_api_key || null,
        binance_api_secret: userData.binance_api_secret || null,
      }),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to connect account');
    }
    
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Get AI predictions (public - no login required)
export async function getPublicPredictions() {
  try {
    const response = await fetch(`${API_BASE_URL}/predictions`);
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Get AI predictions for a specific user
export async function getUserPredictions(username) {
  try {
    const response = await fetch(`${API_BASE_URL}/get_predictions?username=${encodeURIComponent(username)}`);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to get predictions');
    }
    
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Execute a trade
export async function executeTrade(username, symbol, action) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/execute_trade?username=${encodeURIComponent(username)}&symbol=${encodeURIComponent(symbol)}&action=${encodeURIComponent(action)}`,
      {
        method: 'POST',
      }
    );
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to execute trade');
    }
    
    return { success: true, data };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// ===========================================
// EXAMPLE USAGE IN REACT COMPONENTS
// ===========================================

/*
// Example 1: Check Backend Status
import { checkBackendStatus } from './api';

function StatusChecker() {
  const [status, setStatus] = useState(null);
  
  useEffect(() => {
    async function check() {
      const result = await checkBackendStatus();
      setStatus(result);
    }
    check();
  }, []);
  
  return (
    <div>
      {status?.success ? (
        <p>Backend is running: {status.data.message}</p>
      ) : (
        <p>Backend error: {status?.error}</p>
      )}
    </div>
  );
}

// Example 2: Connect Account Form
import { connectAccount } from './api';

function ConnectAccountForm() {
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    const result = await connectAccount({ username });
    
    if (result.success) {
      setMessage(`Success: ${result.data.message}`);
    } else {
      setMessage(`Error: ${result.error}`);
    }
    
    setLoading(false);
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        placeholder="Enter username"
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Connecting...' : 'Connect Account'}
      </button>
      {message && <p>{message}</p>}
    </form>
  );
}

// Example 3: Display AI Predictions
import { getPublicPredictions } from './api';

function PredictionsDisplay() {
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    async function fetchPredictions() {
      const result = await getPublicPredictions();
      if (result.success) {
        setPredictions(result.data);
      }
      setLoading(false);
    }
    fetchPredictions();
  }, []);
  
  if (loading) return <p>Loading predictions...</p>;
  if (!predictions?.success) return <p>Error loading predictions</p>;
  
  return (
    <div>
      <h2>AI Trading Signals</h2>
      <p>Generated at: {predictions.generated_at}</p>
      <div>
        {predictions.signals?.map((signal, index) => (
          <div key={index} style={{ 
            border: '1px solid #ccc', 
            padding: '10px', 
            margin: '10px 0',
            borderRadius: '8px'
          }}>
            <h3>{signal.symbol}</h3>
            <p><strong>Signal:</strong> {signal.signal}</p>
            <p><strong>Confidence:</strong> {signal.confidence}%</p>
            <p><strong>Entry Price:</strong> {signal.entry_price}</p>
            <p><strong>Stop Loss:</strong> {signal.stop_loss}</p>
            <p><strong>Take Profit:</strong> {signal.take_profit}</p>
            <p><strong>Analysis:</strong> {signal.analysis}</p>
            <p><strong>Sentiment:</strong> {signal.market_sentiment}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// Example 4: Execute Trade Button
import { executeTrade } from './api';

function TradeButton({ username, symbol, action }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  
  const handleTrade = async () => {
    setLoading(true);
    const response = await executeTrade(username, symbol, action);
    setResult(response);
    setLoading(false);
  };
  
  return (
    <div>
      <button 
        onClick={handleTrade} 
        disabled={loading}
        style={{
          backgroundColor: action === 'buy' ? '#22c55e' : '#ef4444',
          color: 'white',
          padding: '10px 20px',
          borderRadius: '5px',
          cursor: 'pointer'
        }}
      >
        {loading ? 'Processing...' : `${action.toUpperCase()} ${symbol}`}
      </button>
      {result && (
        <p>{result.success ? result.data.message : result.error}</p>
      )}
    </div>
  );
}
*/
