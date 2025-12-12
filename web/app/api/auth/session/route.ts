import { NextRequest, NextResponse } from 'next/server';
import { createSession, getSession, updateSession, deleteSession } from '@/lib/server-session';

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey, authState } = await request.json();

    if (!appId || !secretKey) {
      return NextResponse.json(
        { error: 'App ID and secret key are required' },
        { status: 400 }
      );
    }

    // Create encrypted session
    await createSession({
      appId,
      secretKey,
      authState,
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Session creation error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const session = await getSession();

    if (!session) {
      return NextResponse.json({ session: null });
    }

    // Return session data without sensitive info
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { secretKey, authState, ...safeSessionData } = session;
    const responseData = {
      session: {
        ...safeSessionData,
        isAuthenticated: session.isAuthenticated
      }
    };

    return NextResponse.json(responseData);
  } catch (error) {
    console.error('Session get error:', error);
    return NextResponse.json({ session: null });
  }
}

export async function PUT(request: NextRequest) {
  try {
    const { accessToken, refreshToken, profile } = await request.json();

    // Update session with tokens
    const updated = await updateSession({
      accessToken,
      refreshToken,
      profile,
    });

    if (!updated) {
      return NextResponse.json(
        { error: 'Invalid session' },
        { status: 401 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Session update error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE() {
  try {
    await deleteSession();
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Session delete error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}