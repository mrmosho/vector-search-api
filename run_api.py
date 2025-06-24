"""
API Runner Module
Responsible for: Starting the API server with command line arguments
"""
import argparse
import uvicorn
import logging

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
    
    if args.reload:
        # For reload mode, use the app factory
        uvicorn.run(
            "api_server:app",
            host=args.host,
            port=args.port,
            reload=True
        )
    else:
        # For production mode, create app directly
        from api_server import APIServer
        api_server = APIServer(csv_path=args.csv)
        app = api_server.get_app()
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=False
        )


if __name__ == "__main__":
    main()