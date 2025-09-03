#!/usr/bin/env python3
"""
Simple HTTP server for hosting your embeddings.
Use this if you want to run your own server instead of using R2.
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import threading
import time

class EmbeddingHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves files from a specific directory."""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        self.directory = os.path.join(os.getcwd(), "embeddings")
        os.makedirs(self.directory, exist_ok=True)
        super().__init__(*args, directory=self.directory, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        # Add CORS headers
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Serve the file
        super().do_GET()
    
    def do_POST(self):
        """Handle POST requests for uploading files."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Extract filename from path
        filename = self.path.strip('/')
        if not filename:
            self.send_error(400, "No filename provided")
            return
        
        # Save the file
        file_path = os.path.join(self.directory, filename)
        try:
            with open(file_path, 'wb') as f:
                f.write(post_data)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success", "message": f"File {filename} uploaded"}).encode())
            
        except Exception as e:
            self.send_error(500, f"Failed to save file: {e}")
    
    def log_message(self, format, *args):
        """Custom logging."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")


def start_server(port: int = 8000, host: str = "0.0.0.0"):
    """Start the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, EmbeddingHandler)
    
    print(f"🌐 Starting HTTP server on http://{host}:{port}")
    print(f"📁 Serving files from: {os.path.join(os.getcwd(), 'embeddings')}")
    print("📝 Upload files via POST to http://yourserver:port/filename")
    print("🔗 Access files via GET at http://yourserver:port/filename")
    print("\n⚠️  Remember to:")
    print("   1. Configure your firewall to allow incoming connections")
    print("   2. Set up SSL/TLS for production use")
    print("   3. Use a reverse proxy like nginx for better performance")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple HTTP server for MANTIS embeddings")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    start_server(args.port, args.host)