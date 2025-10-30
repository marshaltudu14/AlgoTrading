import { NextRequest, NextResponse } from 'next/server';
import { FyersAuthService, FyersAuthError } from '@/lib/fyersAuth';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { accessToken } = body;

    if (!accessToken) {
      return NextResponse.json(
        {
          success: false,
          error: 'Access token is required'
        },
        { status: 400 }
      );
    }

    const authService = new FyersAuthService();
    const isValid = await authService.validateToken(accessToken);

    return NextResponse.json({
      success: true,
      data: {
        valid: isValid
      }
    });

  } catch (error) {
    console.error('Token validation error:', error);

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
        error: 'Internal server error during token validation'
      },
      { status: 500 }
    );
  }
}