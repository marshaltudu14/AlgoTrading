import axios from 'axios';
import { CandleData, Timeframe } from '@/types/fyers';

export type { Timeframe };

export class MarketDataService {
  private apiBase: string;
  private appId: string;

  constructor() {
    this.apiBase = 'https://api-t1.fyers.in';
    this.appId = process.env.FYERS_APP_ID!;
  }

  async getHistoricalData(
    accessToken: string,
    symbol: string,
    timeframe: Timeframe,
    fromDate: string,
    toDate: string
  ): Promise<CandleData[]> {
    try {
      const url = `${this.apiBase}/api/v3/history`;
      const payload = {
        symbol: symbol,
        resolution: timeframe,
        date_format: "1",
        range_from: fromDate,
        range_to: toDate,
        cont_flag: "1"
      };

      const response = await axios.post(url, payload, {
        headers: {
          'Authorization': `${this.appId}:${accessToken}`
        }
      });

      if (response.data.s !== 'ok' || response.data.code !== 200) {
        throw new Error(`Failed to fetch historical data: ${response.data.message}`);
      }

      // Transform the response data to our format
      const candles = response.data.candles || [];
      return candles.map((candle: number[]) => ({
        timestamp: candle[0],
        open: candle[1],
        high: candle[2],
        low: candle[3],
        close: candle[4],
        volume: candle[5]
      }));

    } catch (error) {
      console.error('Error fetching historical data:', error);
      throw error;
    }
  }

  async getQuote(accessToken: string, symbol: string): Promise<Record<string, unknown>> {
    try {
      const url = `${this.apiBase}/api/v2/quotes`;
      const params = { symbols: symbol };

      const response = await axios.get(url, {
        params,
        headers: {
          'Authorization': `${this.appId}:${accessToken}`
        }
      });

      if (response.data.s !== 'ok' || response.data.code !== 200) {
        throw new Error(`Failed to fetch quote: ${response.data.message}`);
      }

      return response.data.d || [];

    } catch (error) {
      console.error('Error fetching quote:', error);
      throw error;
    }
  }

  async getMarketStatus(accessToken: string): Promise<Record<string, unknown>> {
    try {
      const url = `${this.apiBase}/api/v2/market-status`;

      const response = await axios.get(url, {
        headers: {
          'Authorization': `${this.appId}:${accessToken}`
        }
      });

      return response.data;

    } catch (error) {
      console.error('Error fetching market status:', error);
      throw error;
    }
  }

  // Common symbol mappings
  static readonly SYMBOLS = {
    NIFTY: 'NSE:NIFTY50-INDEX',
    BANKNIFTY: 'NSE:NIFTYBANK-INDEX',
    SENSEX: 'BSE:SENSEX-INDEX',
    RELIANCE: 'NSE:RELIANCE-EQ',
    TCS: 'NSE:TCS-EQ',
    HDFCBANK: 'NSE:HDFCBANK-EQ',
  };

  // Helper function to format date for API
  static formatDate(date: Date): string {
    return date.toISOString().split('T')[0];
  }

  // Helper function to get date range for historical data
  static getDateRange(daysBack: number): { fromDate: string; toDate: string } {
    const toDate = new Date();
    const fromDate = new Date();
    fromDate.setDate(toDate.getDate() - daysBack);

    return {
      fromDate: this.formatDate(fromDate),
      toDate: this.formatDate(toDate)
    };
  }
}