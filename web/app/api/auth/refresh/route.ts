import { NextRequest, NextResponse } from 'next/server';
import { FyersAuthService, FyersAuthError } from '@/lib/fyersAuth';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { refreshToken } = body;

    if (!refreshToken) {
      return NextResponse.json(
        {
          success: false,
          error: 'Refresh token is required'
        },
        { status: 400 }
      );
    }

    const authService = new FyersAuthService();
    const newAccessToken = await authService.refreshToken(refreshToken);

    return NextResponse.json({
      success: true,
      data: {
        access_token: newAccessToken
      }
    });

  } catch (error) {
    console.error('Token refresh error:', error);

    if (error instanceof FyersAuthError) {
      return NextResponse.json(
        {
          success: false,
          error: error.message,
          code: error.code || 500
        },
        { status: 400 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error during token refresh'
      },
      { status: 500 }
    );
  }
}