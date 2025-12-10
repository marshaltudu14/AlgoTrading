import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { FYERS_ENDPOINTS } from '@/lib/constants';

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey, refresh_token, pin } = await request.json();

    if (!appId || !secretKey || !refresh_token) {
      return NextResponse.json(
        { error: 'App ID, Secret Key, and Refresh Token are required' },
        { status: 400 }
      );
    }

    // Create SHA-256 hash of app_id:app_secret
    const appIdHash = crypto
      .createHash('sha256')
      .update(`${appId}:${secretKey}`)
      .digest('hex');

    // Exchange refresh_token for new access_token
    const refreshResponse = await fetch(FYERS_ENDPOINTS.VALIDATE_REFRESH_TOKEN, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        appIdHash: appIdHash,
        refresh_token: refresh_token,
        pin: pin || undefined,
      }),
    });

    const data = await refreshResponse.json();

    if (data.s !== 'ok') {
      return NextResponse.json(
        { error: data.message || 'Failed to refresh token' },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      access_token: data.access_token,
    });
  } catch (error) {
    console.error('Error refreshing token:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}