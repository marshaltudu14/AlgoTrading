import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import CallbackClient from './CallbackClient';

export default async function CallbackPage() {
  const cookieStore = await cookies();
  const hasAccessToken = cookieStore.get('fyers_access_token')?.value;

  if (hasAccessToken) {
    redirect('/dashboard');
  }

  return <CallbackClient />;
}