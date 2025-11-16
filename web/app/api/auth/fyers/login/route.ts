import { NextRequest, NextResponse } from 'next/server';
import { apiCall, API_ENDPOINTS, APIError } from '@/lib/api';

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey, redirectUrl, fyersId, pin, totpSecret } = await request.json();

    // Validate required fields
    if (!appId || !secretKey || !redirectUrl || !fyersId || !pin || !totpSecret) {
      return NextResponse.json(
        { error: 'All fields are required' },
        { status: 400 }
      );
    }

    // Call FastAPI backend using centralized API utility
    try {
      const data = await apiCall<{ access_token: string; profile: unknown }>(API_ENDPOINTS.AUTH.FYERS_LOGIN, {
        method: 'POST',
        body: JSON.stringify({
          app_id: appId,
          secret_key: secretKey,
          redirect_uri: redirectUrl,
          fy_id: fyersId,
          pin: pin,
          totp_secret: totpSecret
        }),
      });

      return NextResponse.json({
        success: true,
        access_token: data.access_token,
        profile: data.profile
      });

    } catch (fetchError) {
      console.error('Backend connection error:', fetchError);

      return NextResponse.json(
        {
          error: fetchError instanceof Error ? fetchError.message : 'Authentication service unavailable'
        },
        { status: fetchError instanceof Error && 'status' in fetchError ? (fetchError as APIError).status : 503 }
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