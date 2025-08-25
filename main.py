# Import and run the MCP server
from BA_server import app, server
import os
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    port = int(os.getenv("PORT", "8893"))  # Render provides PORT

    # Run the FastAPI app directly
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
