# AlgoTrading Project

A full-stack algorithmic trading application with Python backend and Next.js frontend.

## Project Structure

```
AlgoTrading/
├── backend/              # Python backend services
│   ├── src/             # Source code modules
│   ├── tests/           # Test suite
│   ├── config/          # Configuration files
│   ├── data/            # Data storage
│   ├── docs/            # Documentation
│   ├── venv/            # Python virtual environment
│   ├── requirements.txt # Python dependencies
│   └── *.py             # Main Python scripts
├── web/                 # Next.js frontend application
│   ├── app/             # Next.js app directory
│   ├── public/          # Static assets
│   ├── package.json     # Node.js dependencies
│   └── ...              # Other Next.js files
└── web-bundles/         # Web development bundles and tools
```

## Getting Started

### Backend (Python)

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend (Next.js)

See [web/README.md](./web/README.md) for detailed instructions.

1. Navigate to the web directory:
   ```bash
   cd web
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000) to view the application.