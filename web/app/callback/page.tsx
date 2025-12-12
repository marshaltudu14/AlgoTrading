import CallbackClient from './CallbackClient';

export default function CallbackPage() {
  // Let the client component handle auth status checking
  // Server-side redirect will be handled by middleware if needed
  return <CallbackClient />;
}