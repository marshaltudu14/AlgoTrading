import { NextResponse } from 'next/server';

export async function POST() {
  try {
    const { deleteSession } = await import('@/lib/server-session');

    // Delete session
    await deleteSession();

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Logout error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}