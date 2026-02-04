# AutoClean: Web-First Data Cleaning Pipeline

AutoClean is a powerful, automated tool for cleaning tabular datasets. Originally a Python CLI utility, it is now a full-stack web application powered by **Next.js** vs **FastAPI**.

## Features

- **Drag & Drop Interface**: Upload CSV or Parquet files instantly.
- **Smart Configuration**:
  - Impute missing values (Mean, Median, Constant).
  - Handle outliers (IQR Winsorization/Clipping).
  - Advanced Scaling (Robust, Standard, MinMax).
  - Categorical Encoding (One-Hot, Ordinal).
- **Secure & Fast**: Processing happens via Python Serverless Functions.
- **Premium UI**: Built with React, TailwindCSS, and Framer Motion.

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.9+

### Installation

1. Install Frontend Dependencies:
   ```bash
   npm install
   ```

2. Install Backend Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

You need to run both the Frontend and the Backend servers.

**Terminal 1 (Backend):**
Start the FastAPI server on port 8000.
```powershell
uvicorn api.index:app --reload --port 8000
```
*Note: Make sure you are in the root directory.*

**Terminal 2 (Frontend):**
Start the Next.js development server.
```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to use the app.

## Deployment to Vercel

This project is optimized for **Vercel**.

1. Push this repository to GitHub.
2. Import the project in Vercel.
3. Vercel will automatically detect the Next.js app and the Python API.
4. **Important**: Ensure `requirements.txt` is in the root (it is by default).

## Project Structure

- `app/`: Next.js Frontend (Pages and Layouts).
- `components/`: React UI Components.
- `api/`: Python Backend (FastAPI handler and logic).
  - `index.py`: API Entry point.
  - `cleaning_pipeline.py`: Core logic.
- `notebooks/` & `scripts/`: Original utility scripts (kept for reference).

## Authors

- **Athun Sujith** - [GitHub](https://github.com/AthunSujith)
