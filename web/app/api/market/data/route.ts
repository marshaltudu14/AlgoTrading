import { NextRequest, NextResponse } from 'next/server';
import { MarketDataService } from '@/lib/marketData';
import { Timeframe } from '@/types/fyers';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get access token from Authorization header or query param
    const authHeader = request.headers.get('authorization');
    const accessToken = searchParams.get('accessToken') || authHeader?.replace('Bearer ', '');

    if (!accessToken) {
      return NextResponse.json(
        {
          success: false,
          error: 'Access token is required'
        },
        { status: 401 }
      );
    }

    // Get query parameters
    const symbol = searchParams.get('symbol');
    const timeframe = searchParams.get('timeframe') as Timeframe || '1';
    const daysBack = parseInt(searchParams.get('daysBack') || '30');

    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          error: 'Symbol is required'
        },
        { status: 400 }
      );
    }

    // Validate timeframe
    const validTimeframes: Timeframe[] = ['1', '3', '5', '10', '15', '30', '60', '120', '240', 'D'];
    if (!validTimeframes.includes(timeframe)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`
        },
        { status: 400 }
      );
    }

    const marketService = new MarketDataService();

    // Get date range
    const { fromDate, toDate } = MarketDataService.getDateRange(daysBack);

    // Fetch historical data
    const data = await marketService.getHistoricalData(
      accessToken,
      symbol,
      timeframe,
      fromDate,
      toDate
    );

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        fromDate,
        toDate,
        candles: data
      }
    });

  } catch (error) {
    console.error('Market data API error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch market data'
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { accessToken, symbol, timeframe = '1', daysBack = 30 } = body;

    if (!accessToken) {
      return NextResponse.json(
        {
          success: false,
          error: 'Access token is required'
        },
        { status: 401 }
      );
    }

    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          error: 'Symbol is required'
        },
        { status: 400 }
      );
    }

    // Validate timeframe
    const validTimeframes: Timeframe[] = ['1', '3', '5', '10', '15', '30', '60', '120', '240', 'D'];
    if (!validTimeframes.includes(timeframe)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`
        },
        { status: 400 }
      );
    }

    const marketService = new MarketDataService();

    // Get date range
    const { fromDate, toDate } = MarketDataService.getDateRange(daysBack);

    // Fetch historical data
    const data = await marketService.getHistoricalData(
      accessToken,
      symbol,
      timeframe,
      fromDate,
      toDate
    );

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        fromDate,
        toDate,
        candles: data
      }
    });

  } catch (error) {
    console.error('Market data API error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch market data'
      },
      { status: 500 }
    );
  }
}