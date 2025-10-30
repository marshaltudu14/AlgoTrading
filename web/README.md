# AlgoTrading Platform

A modern, professional trading platform built with Next.js 16, TypeScript, and Fyers API integration. Features real-time candlestick charts, secure authentication, and a responsive dark-themed interface.

## Features

- 🔐 **Secure Authentication**: Complete Fyers API integration with TOTP support
- 📊 **Real-time Charts**: TradingView-style candlestick charts using lightweight-charts
- 🎨 **Modern UI**: Dark theme with responsive design
- ⚡ **Fast Performance**: Built with Next.js 16 and Zustand for state management
- 🔄 **Auto-refresh**: Market data updates every 30 seconds
- 📱 **Mobile Responsive**: Works seamlessly on all devices

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript
- **State Management**: Zustand with persistence
- **Charts**: Lightweight Charts (TradingView alternative)
- **Styling**: Tailwind CSS v4
- **API Integration**: Fyers Trading API
- **Authentication**: JWT tokens with refresh capability

## Quick Start

### Prerequisites

- Node.js 18+
- Fyers API credentials (APP_ID, SECRET_KEY, etc.)

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   ```

   Edit `.env.local` with your Fyers credentials:
   ```env
   FYERS_APP_ID="YOUR_APP_ID"
   FYERS_SECRET_KEY="YOUR_SECRET_KEY"
   FYERS_REDIRECT_URI="YOUR_REDIRECT_URI"
   FYERS_USER="YOUR_FYERS_USER_ID"
   FYERS_PIN="YOUR_PIN"
   FYERS_TOTP="YOUR_TOTP_SECRET"
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser and navigate to:**
   ```
   http://localhost:3000
   ```

## Project Structure

```
web/
├── app/
│   ├── (auth)/login/          # Login page and layout
│   ├── dashboard/             # Main trading dashboard
│   ├── api/auth/              # Authentication API routes
│   ├── api/market/            # Market data API routes
│   └── layout.tsx             # Root layout with error boundary
├── components/
│   ├── charts/                # Chart components
│   ├── auth/                  # Authentication components
│   └── ui/                    # Reusable UI components
├── stores/                    # Zustand state management
├── lib/                       # Utility functions and API services
├── types/                     # TypeScript type definitions
└── middleware.ts              # Route protection middleware
```

## Available Features

### Chart Features
- Multiple timeframes (1m to 1D)
- Volume overlay
- Interactive controls (pan, zoom, crosshair)
- Auto-refresh every 30 seconds
- Multiple symbol support

### Authentication Features
- TOTP countdown timer
- Auto token refresh
- Secure credential handling
- Persistent login state
- Error handling and validation

### Trading Symbols
- NIFTY 50 Index
- NIFTY BANK Index
- SENSEX Index
- Major stocks (RELIANCE, TCS, HDFC Bank)

## Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

## Security Notes

- 🔒 API credentials are server-side only (never exposed to client)
- 🍪 JWT tokens stored securely with auto-refresh
- 🛡️ Input validation and sanitization
- ⚡ Rate limiting considerations for API calls

## Troubleshooting

### Common Issues

1. **Authentication fails**: Check your Fyers credentials in `.env.local`
2. **Chart not loading**: Ensure you have a valid access token
3. **TOTP issues**: Make sure your TOTP secret is correct and synchronized
4. **API errors**: Check network connectivity and Fyers API status

### Debug Mode

For development, the login form pre-fills with sample credentials when `NODE_ENV=development`.

---

**Built with ❤️ using Next.js, TypeScript, and modern web technologies.**
