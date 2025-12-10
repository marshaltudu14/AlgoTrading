import { NextRequest, NextResponse } from 'next/server';
import { FYERS_ENDPOINTS, COOKIE_NAMES } from '@/lib/constants';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string; timeframe: string }> }
) {
  try {
    const { symbol, timeframe } = await params;
    const { searchParams } = new URL(request.url);

    // Get access token and app_id from cookies
    const cookieStore = request.cookies;
    const accessToken = cookieStore.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
    const appId = cookieStore.get(COOKIE_NAMES.APP_ID)?.value;

    if (!accessToken || !appId) {
      return NextResponse.json(
        { error: 'Not authenticated' },
        { status: 401 }
      );
    }

    // Get days from query params or find from timeframe config
    let days = parseInt(searchParams.get('days') || '15');

      // Default to 15 days if not specified
    if (!searchParams.has('days')) {
      days = 15;
    }

    // Calculate date range
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);

    // Format dates as YYYY-MM-DD
    const rangeFrom = startDate.toISOString().split('T')[0];
    const rangeTo = endDate.toISOString().split('T')[0];

        // Use timeframe directly as Fyers resolution
    const resolution = timeframe;

    // Build the URL
    const url = new URL(FYERS_ENDPOINTS.HISTORY);
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('resolution', resolution);
    url.searchParams.set('date_format', '1');
    url.searchParams.set('range_from', rangeFrom);
    url.searchParams.set('range_to', rangeTo);
    url.searchParams.set('cont_flag', '1');

    // Make request to Fyers API
    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `${appId}:${accessToken}`,
      },
    });

    if (!response.ok) {
      throw new Error(`Fyers API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (data.s !== 'ok') {
      throw new Error(data.message || 'Fyers API returned error');
    }

    // Transform data to match expected format and remove duplicates
    const candles = data.candles || [];
    const dataMap = new Map<number, [number, number, number, number, number]>();

    // Use a Map to automatically handle duplicates
    candles.forEach((candle: number[]) => {
      const timestamp = candle[0];
      // Only keep the first occurrence of each timestamp
      if (!dataMap.has(timestamp)) {
        dataMap.set(timestamp, [
          candle[1], // open
          candle[2], // high
          candle[3], // low
          candle[4], // close
          candle[5] || 0, // volume
        ]);
      }
    });

    // Convert Map back to array and sort by timestamp
    const uniqueData = Array.from(dataMap.entries())
      .map(([timestamp, [open, high, low, close, volume]]) => ({
        timestamp,
        open,
        high,
        low,
        close,
        volume,
      }))
      .sort((a, b) => a.timestamp - b.timestamp);

    return NextResponse.json({
      success: true,
      data: uniqueData,
    });

  } catch (error) {
    console.error('Error fetching candle data:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch candle data'
      },
      { status: 500 }
    );
  }
}