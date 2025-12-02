// ===========================================
// LOVABLE FRONTEND INTEGRATION CODE (TypeScript)
// Copy and use this code in your Lovable app
// ===========================================

// Your Backend URL - UPDATE THIS if it changes
const API_BASE_URL = 'https://2d09f74f-9771-4c85-b6d6-a965d884c276-00-3q1c8s2ku1arv.kirk.replit.dev';

// ===========================================
// TYPE DEFINITIONS
// ===========================================

interface UserAccount {
  username: string;
  mt5_login?: string;
  mt5_password?: string;
  mt5_server?: string;
  binance_api_key?: string;
  binance_api_secret?: string;
}

interface TradingSignal {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  entry_price: string;
  stop_loss: string;
  take_profit: string;
  risk_reward_ratio: string;
  timeframe: 'short-term' | 'medium-term' | 'long-term';
  analysis: string;
  market_sentiment: 'bullish' | 'bearish' | 'neutral';
}

interface PredictionsResponse {
  success: boolean;
  generated_at?: string;
  model?: string;
  signals?: TradingSignal[];
  total_signals?: number;
  error?: string;
  details?: string;
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// ===========================================
// API SERVICE FUNCTIONS
// ===========================================

// Check if backend is running
export async function checkBackendStatus(): Promise<ApiResponse<{ message: string }>> {
  try {
    const response = await fetch(`${API_BASE_URL}/`);
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

// Connect a new user account
export async function connectAccount(userData: UserAccount): Promise<ApiResponse<{ status: string; message: string }>> {
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
    return { success: false, error: (error as Error).message };
  }
}

// Get AI predictions (public - no login required)
export async function getPublicPredictions(): Promise<ApiResponse<PredictionsResponse>> {
  try {
    const response = await fetch(`${API_BASE_URL}/predictions`);
    const data = await response.json();
    return { success: true, data };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

// Get AI predictions for a specific user
export async function getUserPredictions(username: string): Promise<ApiResponse<{
  username: string;
  request_time: string;
  predictions: PredictionsResponse;
}>> {
  try {
    const response = await fetch(`${API_BASE_URL}/get_predictions?username=${encodeURIComponent(username)}`);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.detail || 'Failed to get predictions');
    }
    
    return { success: true, data };
  } catch (error) {
    return { success: false, error: (error as Error).message };
  }
}

// Execute a trade
export async function executeTrade(
  username: string, 
  symbol: string, 
  action: 'buy' | 'sell'
): Promise<ApiResponse<{ status: string; message: string }>> {
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
    return { success: false, error: (error as Error).message };
  }
}

// ===========================================
// REACT HOOKS (Optional - for easier usage)
// ===========================================

/*
import { useState, useEffect } from 'react';

// Hook to fetch predictions
export function usePredictions() {
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);
    
    const result = await getPublicPredictions();
    
    if (result.success && result.data) {
      setPredictions(result.data);
    } else {
      setError(result.error || 'Unknown error');
    }
    
    setLoading(false);
  };
  
  useEffect(() => {
    fetchPredictions();
  }, []);
  
  return { predictions, loading, error, refresh: fetchPredictions };
}

// Hook to manage user account
export function useAccount() {
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const connect = async (userData: UserAccount) => {
    setLoading(true);
    setError(null);
    
    const result = await connectAccount(userData);
    
    if (result.success) {
      setUsername(userData.username);
    } else {
      setError(result.error || 'Failed to connect');
    }
    
    setLoading(false);
    return result;
  };
  
  return { username, loading, error, connect };
}

// Hook to execute trades
export function useTrade() {
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<ApiResponse<{ status: string; message: string }> | null>(null);
  
  const trade = async (username: string, symbol: string, action: 'buy' | 'sell') => {
    setLoading(true);
    const result = await executeTrade(username, symbol, action);
    setLastResult(result);
    setLoading(false);
    return result;
  };
  
  return { loading, lastResult, trade };
}
*/
