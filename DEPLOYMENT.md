# AlgoTrading Vercel Deployment Guide

This guide explains how to deploy the AlgoTrading application to Vercel with both Next.js frontend and Python FastAPI backend.

## Project Structure

```
AlgoTrading/
├── frontend/          # Next.js application
├── backend/           # FastAPI application
├── api/              # Vercel serverless functions
│   ├── index.py      # Main API entry point
│   └── requirements.txt # Python dependencies
├── src/              # Shared Python modules
├── vercel.json       # Vercel configuration
└── .vercelignore     # Files to exclude from deployment
```

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm i -g vercel`
3. **Git Repository**: Push your code to GitHub, GitLab, or Bitbucket

## Deployment Steps

### 1. Prepare for Deployment

Ensure all TypeScript errors and ESLint warnings are fixed:

```bash
cd frontend
npx tsc --noEmit
npm run lint
```

### 2. Deploy to Vercel

#### Option A: Deploy via Vercel Dashboard
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your Git repository
4. Vercel will automatically detect the configuration from `vercel.json`
5. Click "Deploy"

#### Option B: Deploy via CLI
```bash
# Login to Vercel
vercel login

# Deploy from project root
vercel

# Follow the prompts:
# - Set up and deploy? Yes
# - Which scope? (select your account)
# - Link to existing project? No
# - Project name? (default or custom name)
# - Directory? ./
```

### 3. Environment Variables

Set up environment variables in Vercel Dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add any required environment variables:
   - `NEXT_PUBLIC_API_URL` (optional, defaults to relative URLs)
   - Any other environment variables your backend needs

## Configuration Details

### vercel.json Configuration

The `vercel.json` file configures:
- **Next.js Frontend**: Built from `frontend/` directory
- **Python Backend**: Serverless functions from `api/` directory
- **Routing**: API calls to `/api/*` go to Python backend, everything else to Next.js
- **Python Runtime**: Python 3.9 with 30-second timeout

### API Structure

The Python backend is deployed as serverless functions:
- **Entry Point**: `api/index.py`
- **Dependencies**: `api/requirements.txt` (lightweight subset)
- **Fallback**: Minimal API if full backend can't load

### CORS Configuration

The backend is configured to accept requests from:
- `localhost:3000` and `localhost:3001` (development)
- `*.vercel.app` domains (production)

## Local Development

For local development, continue using:

```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
cd frontend
npm run dev
```

## Production URLs

After deployment, your application will be available at:
- **Frontend**: `https://your-project-name.vercel.app`
- **API**: `https://your-project-name.vercel.app/api/health`

## Troubleshooting

### Common Issues

1. **Build Failures**: Check the build logs in Vercel dashboard
2. **API Errors**: Verify Python dependencies in `api/requirements.txt`
3. **CORS Issues**: Ensure your domain is in the CORS allow list
4. **Large Dependencies**: Some ML libraries may be too large for serverless

### Limitations

- **File System**: Serverless functions have limited file system access
- **Memory**: Limited to 1GB RAM per function
- **Timeout**: 30-second maximum execution time
- **Cold Starts**: First request may be slower

### Monitoring

Monitor your deployment:
- **Vercel Dashboard**: View deployment status and logs
- **Function Logs**: Check serverless function execution logs
- **Analytics**: Monitor performance and usage

## Support

For issues:
1. Check Vercel documentation: [vercel.com/docs](https://vercel.com/docs)
2. Review deployment logs in Vercel dashboard
3. Test API endpoints individually
