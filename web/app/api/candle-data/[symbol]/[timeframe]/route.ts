import { NextRequest, NextResponse } from 'next/server';
import { FYERS_ENDPOINTS, COOKIE_NAMES } from '@/lib/constants';

interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DateRange {
  startDate: Date;
  endDate: Date;
}

// Helper function to split date range into chunks of max 100 days
function splitDateRange(startDate: Date, endDate: Date): DateRange[] {
  const ranges: DateRange[] = [];
  const currentStart = new Date(startDate);

  while (currentStart < endDate) {
    const currentEnd = new Date(currentStart);
    // Add 99 days to stay under the 100-day limit (inclusive counting)
    currentEnd.setDate(currentEnd.getDate() + 99);

    // If this chunk goes beyond the overall end date, cap it
    if (currentEnd > endDate) {
      ranges.push({
        startDate: currentStart,
        endDate: endDate
      });
      break;
    }

    ranges.push({
      startDate: currentStart,
      endDate: currentEnd
    });

    // Move to the next chunk (day after current end)
    currentStart.setDate(currentStart.getDate() + 100);
  }

  return ranges;
}

// Helper function to fetch data for a specific date range
async function fetchCandleData(
  url: URL,
  appId: string,
  accessToken: string,
  rangeFrom: string,
  rangeTo: string
): Promise<CandleData[]> {
  const searchParams = new URLSearchParams(url.searchParams);
  searchParams.set('range_from', rangeFrom);
  searchParams.set('range_to', rangeTo);

  const fetchUrl = new URL(url.origin + url.pathname);
  fetchUrl.search = searchParams.toString();

  const response = await fetch(fetchUrl.toString(), {
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

  const candles = data.candles || [];

  // Transform data to match expected format
  return candles.map((candle: number[]) => ({
    timestamp: candle[0],
    open: candle[1],
    high: candle[2],
    low: candle[3],
    close: candle[4],
    volume: candle[5] || 0,
  }));
}

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

    // Get days from query params or default to 15 days
    let days = parseInt(searchParams.get('days') || '15');

    if (!searchParams.has('days')) {
      days = 15;
    }

    // Calculate overall date range
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(endDate.getDate() - days);

    // Use timeframe directly as Fyers resolution
    const resolution = timeframe;

    // Build the base URL (without range dates)
    const url = new URL(FYERS_ENDPOINTS.HISTORY);
    url.searchParams.set('symbol', symbol);
    url.searchParams.set('resolution', resolution);
    url.searchParams.set('date_format', '1');
    url.searchParams.set('cont_flag', '1');

    let allCandleData: CandleData[] = [];

    // If days <= 100, fetch in single request (maintain backward compatibility)
    if (days <= 100) {
      const rangeFrom = startDate.toISOString().split('T')[0];
      const rangeTo = endDate.toISOString().split('T')[0];

      const candleData = await fetchCandleData(url, appId, accessToken, rangeFrom, rangeTo);
      allCandleData = candleData;
    } else {
      // For > 100 days, split into multiple chunks
      const dateRanges = splitDateRange(startDate, endDate);

      console.log(`Fetching data for ${days} days in ${dateRanges.length} chunks`);

      for (let i = 0; i < dateRanges.length; i++) {
        const range = dateRanges[i];
        const rangeFrom = range.startDate.toISOString().split('T')[0];
        const rangeTo = range.endDate.toISOString().split('T')[0];

        console.log(`Fetching chunk ${i + 1}/${dateRanges.length}: ${rangeFrom} to ${rangeTo}`);

        try {
          const candleData = await fetchCandleData(url, appId, accessToken, rangeFrom, rangeTo);
          allCandleData = allCandleData.concat(candleData);

          // Add a small delay between requests to avoid rate limiting
          if (i < dateRanges.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 200)); // 200ms delay
          }
        } catch (error) {
          console.error(`Error fetching chunk ${i + 1}:`, error);
          throw new Error(`Failed to fetch data for range ${rangeFrom} to ${rangeTo}`);
        }
      }
    }

    // Remove duplicates and sort by timestamp
    const dataMap = new Map<number, CandleData>();

    // Use a Map to automatically handle duplicates
    allCandleData.forEach(candle => {
      if (!dataMap.has(candle.timestamp)) {
        dataMap.set(candle.timestamp, candle);
      }
    });

    // Convert Map back to array and sort by timestamp
    const uniqueData = Array.from(dataMap.values())
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