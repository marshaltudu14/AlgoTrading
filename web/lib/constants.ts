// FYERS API Constants
export const FYERS_API_BASE = 'https://api-t1.fyers.in/api/v3';

// FYERS API Endpoints
export const FYERS_ENDPOINTS = {
  GENERATE_AUTHCODE: `${FYERS_API_BASE}/generate-authcode`,
  VALIDATE_AUTHCODE: `${FYERS_API_BASE}/validate-authcode`,
  VALIDATE_REFRESH_TOKEN: `${FYERS_API_BASE}/validate-refresh-token`,
  LOGOUT: `${FYERS_API_BASE}/logout`,
  PROFILE: `${FYERS_API_BASE}/profile`,
  FUNDS: `${FYERS_API_BASE}/funds`,
  HOLDINGS: `${FYERS_API_BASE}/holdings`,
  ORDERS: `${FYERS_API_BASE}/orders`,
  POSITIONS: `${FYERS_API_BASE}/positions`,
  TRADEBOOK: `${FYERS_API_BASE}/tradebook`,
  HISTORY: 'https://api-t1.fyers.in/data/history',
} as const;