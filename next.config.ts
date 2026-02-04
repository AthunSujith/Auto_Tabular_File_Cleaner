import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    /**
     * In development: Proxy /api requests to the local Python FastApi server (port 8000).
     * In production: Vercel automatically routes /api to the python function.
     */
    return process.env.NODE_ENV === 'development'
      ? [
        {
          source: '/api/:path*',
          destination: 'http://127.0.0.1:8000/api/:path*',
        },
      ]
      : [];
  },
};

export default nextConfig;
