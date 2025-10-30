// Fyers API Types

export interface FyersCredentials {
  app_id: string;
  secret_key: string;
  redirect_uri: string;
  fy_id: string;
  pin: string;
  totp_secret: string;
}

export interface FyersUser {
  name: string;
  email: string;
  mobile: string;
  capital: number;
  available_balance: number;
  profile_data: {
    name: string;
    display_name?: string;
    fy_id: string;
    email_id: string;
    pan?: string;
    mobile_number: string;
    totp: boolean;
    pwd_to_expire?: number;
    ddpi_enabled: boolean;
    mtf_enabled: boolean;
  };
  funds_data: Array<{
    id: number;
    title: string;
    equityAmount: number;
    commodityAmount: number;
  }>;
}

export interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketDataResponse {
  s: string; // "ok" | "error"
  code: number;
  message: string;
  data?: CandleData[];
}

export interface AuthResponse {
  s: string; // "ok" | "error"
  code: number;
  message: string;
  access_token?: string;
  refresh_token?: string;
}

export interface ProfileResponse {
  s: string;
  code: number;
  message: string;
  data?: FyersUser['profile_data'];
}

export interface FundsResponse {
  s: string;
  code: number;
  message: string;
  fund_limit?: Array<{
    id: number;
    title: string;
    equityAmount: number;
    commodityAmount: number;
  }>;
}

export interface OrderRequest {
  symbol: string;
  qty: number;
  type: 1 | 2 | 3 | 4; // Limit | Market | Stop | StopLimit
  side: 1 | -1; // Buy | Sell
  productType: "CNC" | "INTRADAY" | "MARGIN" | "CO" | "BO" | "MTF";
  limitPrice: number;
  stopPrice: number;
  validity: "DAY" | "IOC";
  disclosedQty: number;
  offlineOrder: boolean;
  stopLoss: number;
  takeProfit: number;
  orderTag?: string;
}

export interface OrderResponse {
  s: string;
  code: number;
  message: string;
  id?: string;
}

export interface Position {
  symbol: string;
  id: string;
  buyAvg: number;
  buyQty: number;
  sellAvg: number;
  sellQty: number;
  netAvg: number;
  netQty: number;
  side: 1 | -1;
  productType: string;
  realized_profit: number;
  pl: number;
  ltp: number;
}

export interface PositionsResponse {
  s: string;
  code: number;
  message: string;
  netPositions?: Position[];
  overall?: {
    count_total: number;
    count_open: number;
    pl_total: number;
    pl_realized: number;
    pl_unrealized: number;
  };
}

// Timeframe options for historical data
export type Timeframe = "1" | "3" | "5" | "10" | "15" | "30" | "60" | "120" | "240" | "D";

// Chart types
export interface ChartDataPoint {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
  value?: number; // For volume
}

export interface InstrumentInfo {
  symbol: string;
  name: string;
  exchange: string;
  segment: string;
  fytoken: string;
  lot_size: number;
  tick_size: number;
}