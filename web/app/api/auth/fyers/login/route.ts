import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey, redirectUrl, fyersId, pin } = await request.json();

    // Validate required fields
    if (!appId || !secretKey || !redirectUrl || !fyersId || !pin) {
      return NextResponse.json(
        { error: 'All fields are required' },
        { status: 400 }
      );
    }

    // Call FastAPI backend
    try {
      const response = await fetch('http://localhost:8000/auth/fyers/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          app_id: appId,
          secret_key: secretKey,
          redirect_uri: redirectUrl,
          fy_id: fyersId,
          pin: pin
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        return NextResponse.json({
          success: true,
          access_token: data.access_token,
          profile: data.profile
        });
      } else {
        return NextResponse.json(
          { error: data.error || 'Authentication failed' },
          { status: response.status || 401 }
        );
      }

    } catch (fetchError) {
      console.error('FastAPI connection error:', fetchError);
      return NextResponse.json(
        { error: 'Unable to connect to authentication service. Please ensure the backend is running on localhost:8000' },
        { status: 503 }
      );
    }

  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}