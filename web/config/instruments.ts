export interface Instrument {
  id: number;
  name: string;
  symbol: string;
  exchangeSymbol: string;
  type: 'index' | 'stock';
  lotSize: number;
  tickSize: number;
}

export interface Timeframe {
  id: number;
  name: string;
  description: string;
  days: number; // Number of past days to fetch data for
}

export const INSTRUMENTS: Instrument[] = [
  {
    id: 0,
    name: "Bank Nifty",
    symbol: "Bank_Nifty",
    exchangeSymbol: "NSE:NIFTYBANK-INDEX",
    type: "index",
    lotSize: 35,
    tickSize: 0.05,
  },
  {
    id: 1,
    name: "Nifty 50",
    symbol: "Nifty",
    exchangeSymbol: "NSE:NIFTY50-INDEX",
    type: "index",
    lotSize: 75,
    tickSize: 0.05,
  },
  {
    id: 2,
    name: "Bankex",
    symbol: "Bankex",
    exchangeSymbol: "NSE:BANKEX-INDEX",
    type: "index",
    lotSize: 30,
    tickSize: 0.05,
  },
  {
    id: 3,
    name: "Finnifty",
    symbol: "Finnifty",
    exchangeSymbol: "NSE:FINNIFTY-INDEX",
    type: "index",
    lotSize: 65,
    tickSize: 0.05,
  },
  {
    id: 4,
    name: "Sensex",
    symbol: "Sensex",
    exchangeSymbol: "BSE:SENSEX-INDEX",
    type: "index",
    lotSize: 20,
    tickSize: 0.05,
  },
  {
    id: 5,
    name: "Reliance Industries",
    symbol: "RELIANCE",
    exchangeSymbol: "NSE:RELIANCE-EQ",
    type: "stock",
    lotSize: 1,
    tickSize: 0.05,
  },
  {
    id: 6,
    name: "Tata Consultancy Services",
    symbol: "TCS",
    exchangeSymbol: "NSE:TCS-EQ",
    type: "stock",
    lotSize: 1,
    tickSize: 0.05,
  },
  {
    id: 7,
    name: "HDFC Bank",
    symbol: "HDFC",
    exchangeSymbol: "NSE:HDFCBANK-EQ",
    type: "stock",
    lotSize: 1,
    tickSize: 0.05,
  },
];

export const TIMEFRAMES: Timeframe[] = [
  { id: 0, name: "1", description: "1 minute", days: 7 },
  { id: 1, name: "2", description: "2 minutes", days: 7 },
  { id: 2, name: "3", description: "3 minutes", days: 10 },
  { id: 3, name: "5", description: "5 minutes", days: 15 },
  { id: 4, name: "10", description: "10 minutes", days: 20 },
  { id: 5, name: "15", description: "15 minutes", days: 30 },
  { id: 6, name: "20", description: "20 minutes", days: 30 },
  { id: 7, name: "30", description: "30 minutes", days: 45 },
  { id: 8, name: "45", description: "45 minutes", days: 60 },
  { id: 9, name: "60", description: "1 hour", days: 90 },
  { id: 10, name: "120", description: "2 hours", days: 100 },
  { id: 11, name: "180", description: "3 hours", days: 100 },
  { id: 12, name: "240", description: "4 hours", days: 100 },
];

export const DEFAULT_INSTRUMENT = INSTRUMENTS[1]; // Nifty 50
export const DEFAULT_TIMEFRAME = TIMEFRAMES[3]; // 5 minutes