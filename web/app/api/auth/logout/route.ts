import { NextRequest, NextResponse } from 'next/server';
import { apiCall, API_ENDPOINTS, getAuthHeaders } from '@/lib/api';

export async function POST(request: NextRequest) {
  try {
    // Get auth token from request headers if available
    const authHeader = request.headers.get('authorization');
    const accessToken = authHeader?.replace('Bearer ', '');

    // Call FastAPI backend logout endpoint using centralized API utility
    try {
      await apiCall(API_ENDPOINTS.AUTH.LOGOUT, {
        method: 'POST',
        headers: getAuthHeaders(accessToken),
      });
    } catch (error) {
      // Log the error but don't fail the logout
      console.log('Backend logout failed, but proceeding with frontend logout:', error);
    }

    // Always return success - logout should work even if backend is down
    return NextResponse.json({
      success: true,
      message: 'Logged out successfully'
    });

  } catch (error) {
    console.error('Logout API error:', error);

    // Always return success for logout - we want frontend to proceed
    return NextResponse.json({
      success: true,
      message: 'Logged out successfully'
    });
  }
}