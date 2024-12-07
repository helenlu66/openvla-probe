from fastapi import FastAPI, WebSocket
import logging

app = FastAPI()

# Add logging to track server activity
logging.basicConfig(level=logging.INFO)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Client connected")
    try:
        while True:
            # Receive a message from the client
            data = await websocket.receive_text()
            logging.info(f"Received message: {data}")

            # Echo the message back to the client
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logging.error(f"WebSocket connection closed: {e}")
    finally:
        logging.info("Client disconnected")
