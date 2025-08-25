# Import and run the MCP server
from BA_server import app
from fastmcp import SseServer
from anyio import run
import os
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    port = int(os.getenv("PORT", "8893"))  # Render provides PORT

    # Mount BOTH GET /sse (event stream) and POST /sse (JSON-RPC)
    transport = SseServer(app, path="/sse")

    run(app.run(transport=transport, host="0.0.0.0", port=port))
