import { NextRequest, NextResponse } from 'next/server';
import { apiCall, API_ENDPOINTS } from '@/lib/api';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string; timeframe: string }> }
) {
  const { symbol, timeframe } = await params;
  try {
    const { searchParams } = new URL(request.url);

    // Extract optional query parameters
    const start_date = searchParams.get('start_date');
    const end_date = searchParams.get('end_date');

    // Build URL with query parameters
    let url = `${API_ENDPOINTS.CANDLE_DATA?.replace('{symbol}', symbol).replace('{timeframe}', timeframe) || `/candle-data/${symbol}/${timeframe}`}`;

    if (start_date || end_date) {
      const params = new URLSearchParams();
      if (start_date) params.append('start_date', start_date);
      if (end_date) params.append('end_date', end_date);
      url += `?${params.toString()}`;
    }

    // Call FastAPI backend using centralized API utility
    const data = await apiCall(url);

    return NextResponse.json(data);

  } catch (error) {
    console.error('Candle data API error:', error);

    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch candle data';

    return NextResponse.json(
      {
        success: false,
        error: errorMessage,
        data: []
      },
      { status: 500 }
    );
  }
}