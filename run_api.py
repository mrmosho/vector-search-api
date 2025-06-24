"""
API Runner Module
Responsible for: Starting the API server with command line arguments
"""
import argparse
import uvicorn
import logging

from .api_server import APIServer

logger = logging.getLogger(__name__)


def main():
    """Main entry point for running the API server"""
    parser = argparse.ArgumentParser(description="Hybrid Search API Server")
    parser.add_argument("--csv", default="data.csv", help="Path to CSV data file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Search API server...")
    logger.info(f"📊 Data file: {args.csv}")
    logger.info(f"🌐 Server: http://{args.host}:{args.port}")
    
    # Create API server
    api_server = APIServer(csv_path=args.csv)
    app = api_server.get_app()
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()