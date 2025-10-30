import { NextResponse } from 'next/server';
import { FyersAuthService, FyersAuthError } from '@/lib/fyersAuth';

export async function POST() {
  try {
    const authService = new FyersAuthService();

    // Attempt to authenticate and get access token
    const accessToken = await authService.authenticate();

    // Get user profile
    const userProfile = await authService.getUserProfile(accessToken);

    // Create response
    const response = NextResponse.json({
      success: true,
      data: {
        access_token: accessToken,
        user: userProfile
      }
    });

    // Set HTTP-only cookie for additional security
    response.cookies.set('fyers_token', accessToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 24 * 60 * 60, // 24 hours
      path: '/'
    });

    return response;

  } catch (error) {
    console.error('Authentication error:', error);

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
        error: 'Internal server error during authentication'
      },
      { status: 500 }
    );
  }
}