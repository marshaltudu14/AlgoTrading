import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { join } from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { processedCandles, symbol, timeframe } = body;

    if (!processedCandles || !Array.isArray(processedCandles)) {
      return NextResponse.json(
        { error: 'Invalid processedCandles data' },
        { status: 400 }
      );
    }

    // Create backend/data directory if it doesn't exist
    const dataDir = join(process.cwd(), '..', 'backend', 'data', 'processed');
    try {
      await mkdir(dataDir, { recursive: true });
    } catch (error) {
      // Directory already exists
    }

    // Generate filename based on symbol and timeframe
    const filename = `features_${symbol.toLowerCase()}_${timeframe.toLowerCase()}_web.csv`;
    const filePath = join(dataDir, filename);

    // Prepare CSV content
    const headers = [
      'timestamp',
      'open',
      'high',
      'low',
      'close',
      'dist_sma_5',
      'dist_sma_10',
      'dist_sma_20',
      'dist_sma_50',
      'dist_sma_100',
      'dist_sma_200',
      'dist_ema_5',
      'dist_ema_10',
      'dist_ema_20',
      'dist_ema_50',
      'dist_ema_100',
      'dist_ema_200',
      'macd_pct',
      'macd_signal_pct',
      'macd_hist_pct',
      'rsi_14',
      'rsi_21',
      'adx',
      'di_plus',
      'di_minus',
      'atr_pct',
      'atr',
      'bb_width_pct',
      'bb_position',
      'trend_slope',
      'trend_strength',
      'trend_direction',
      'price_change_pct',
      'price_change_abs',
      'hl_range_pct',
      'body_size_pct',
      'upper_shadow_pct',
      'lower_shadow_pct',
      'volatility_10',
      'volatility_20'
    ];

    // Convert to CSV format
    const csvRows = [headers.join(',')];

    processedCandles.forEach((candle: any) => {
      const row = headers.map(header => {
        const value = candle[header];
        // Handle undefined/null values
        if (value === undefined || value === null) {
          return '';
        }
        // Convert to string and escape commas
        return String(value).includes(',') ? `"${value}"` : String(value);
      });
      csvRows.push(row.join(','));
    });

    const csvContent = csvRows.join('\n');

    // Write to file
    await writeFile(filePath, csvContent, 'utf8');

    return NextResponse.json({
      success: true,
      message: `Saved ${processedCandles.length} processed candles to ${filename}`,
      filename,
      filePath
    });

  } catch (error) {
    console.error('Error saving processed data:', error);
    return NextResponse.json(
      { error: 'Failed to save processed data' },
      { status: 500 }
    );
  }
}