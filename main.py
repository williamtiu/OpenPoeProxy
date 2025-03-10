#!/usr/bin/env python
"""
OpenAI-Compatible API Server via Poe API Proxy

This FastAPI application receives OpenAI-format requests and proxies them to a Poe API bot.
It supports both normal and streaming responses (SSE). A simple HTML test page is also included.

Author: williamtiu
License: MIT License
"""

import os
import time
import asyncio
import json
import uuid
from typing import List, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fastapi_poe as fp

# =============================================================================
# 1. OpenAI-Compatible API Data Structures
# =============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    choices: List[ChatChoice]
    usage: ChatUsage

# =============================================================================
# 2. FastAPI Initialization and Global Parameters
# =============================================================================
app = FastAPI(
    title="OpenAI Compatible API Server via Poe API",
    description=(
        "A FastAPI server that uses fastapi_poe to interact with the Poe API and "
        "provide an OpenAI-like chat completions endpoint."
    ),
    version="0.1.0"
)

# Enable CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default Poe API key from the environment variable.
# This will be used if an API key is not provided in the request headers.

POE_API_KEY = os.environ.get("POE_API_KEY", "<Other API KEY>")

# =============================================================================
# 3. Helper Functions: Combine Messages & Call Poe API
# =============================================================================
def combine_messages(messages: List[ChatMessage]) -> str:
    """
    Combine multiple chat messages into a single prompt string.
    
    Example:
        user: Hello
        assistant: Hi, how can I help you?
        user: Tell me about yourself.
        
    This function will concatenate the messages into:
        "user: Hello\nassistant: Hi, how can I help you?\nuser: Tell me about yourself."
    """
    prompt = ""
    for msg in messages:
        prompt += f"{msg.role}: {msg.content}\n"
    return prompt.strip()


async def generate_poe_response(
    api_key: str,
    bot_name: str,
    prompt: str
) -> AsyncGenerator[str, None]:
    """
    Retrieve a response from the Poe API using fastapi_poe.

    Yields:
        str: Each chunk of the response text from the Poe API.
    """
    try:
        # Build the ProtocolMessage object for the Poe API.
        message_obj = fp.ProtocolMessage(role="user", content=prompt)
        async for partial in fp.get_bot_response(
            messages=[message_obj],
            bot_name=bot_name,
            api_key=api_key
        ):
            if partial.text:
                yield partial.text
            # Add a short sleep to allow asynchronous processing
            await asyncio.sleep(0.01)
    except Exception as e:
        # Return the error message as part of the stream.
        error_message = e.args[0] if e.args else repr(e)
        yield f"Error: {error_message}"

# =============================================================================
# 4. OpenAI-Compatible Endpoint for Chat Completions
# =============================================================================
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest, request: Request):
    """
    Process a chat completions request compatible with OpenAI API.

    If the `stream` flag is True, a Streaming SSE response is returned,
    otherwise a complete JSON response is returned.
    """
    # Retrieve API key from the 'Authorization' header using "Bearer <key>" format.
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    else:
        api_key = auth_header or POE_API_KEY

    # Use req.model as the Poe bot name.
    bot_name = req.model

    # Combine the list of chatting messages into one prompt.
    prompt = combine_messages(req.messages)
    prompt_tokens = len(prompt.split())

    # --- Streaming Response ---
    if req.stream:
        response_id = "chatcmpl-" + str(uuid.uuid4())
        created_time = int(time.time())

        async def event_generator() -> AsyncGenerator[str, None]:
            # Stream each chunk as an SSE event with JSON data.
            async for chunk in generate_poe_response(api_key, bot_name, prompt):
                event_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": bot_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            # Signal the end of the stream.
            final_event = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": bot_name,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ]
            }
            yield f"data: {json.dumps(final_event)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # --- Non-streaming JSON Response ---
    else:
        full_text = ""
        async for chunk in generate_poe_response(api_key, bot_name, prompt):
            full_text += chunk
        completion_tokens = len(full_text.split())
        response_obj = ChatResponse(
            id="chatcmpl-" + str(int(time.time())),
            object="chat.completion",
            created=int(time.time()),
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        return JSONResponse(content=response_obj.dict())

# =============================================================================
# 5. Additional Endpoint: Streaming Response for Front-End Testing
# =============================================================================
@app.get("/stream-response")
async def stream_response(
    api_key: str = Query(..., description="Poe API Key (or OpenAI API Key)"),
    bot_name: str = Query(..., description="Bot Name (e.g. 'Gemini-2.0-Pro')"),
    message: str = Query(..., description="User message")
):
    """
    A GET endpoint for testing streaming responses directly from a browser.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        prompt = f"user: {message}"
        async for chunk in generate_poe_response(api_key, bot_name, prompt):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# =============================================================================
# 6. Additional Endpoint: Test HTML Page
# =============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OpenAI-Compatible API Server Test</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; line-height: 1.6; }
    textarea, input[type="text"], button { width: 100%; padding: 10px; margin: 5px 0; font-size: 16px; }
    button { background-color: #4CAF50; color: white; border: none; cursor: pointer; }
    button:hover { background-color: #45a049; }
    #response { border: 1px solid #ddd; border-radius: 4px; padding: 15px; background-color: #f9f9f9; min-height: 100px; white-space: pre-wrap; margin-top: 10px; }
    .loading { display: none; margin: 10px 0; padding: 10px; text-align: center; background-color: #f0f0f0; border: 1px solid #ccc; }
  </style>
</head>
<body>
  <h1>OpenAI-Compatible API Server Test Interface</h1>
  <form id="test-form">
    <label for="api-key">API Key (Authorization Header format):</label>
    <input type="text" id="api-key" placeholder="Enter your API Key" required />

    <label for="bot-name">Bot Name (model):</label>
    <input type="text" id="bot-name" placeholder="e.g. Gemini-2.0-Pro" required />

    <label for="message">Your Message:</label>
    <textarea id="message" placeholder="Type your message here..." required></textarea>

    <label>
      <input type="checkbox" id="stream-toggle" />
      Use Streaming Response (SSE)
    </label>
    <button type="submit">Send Message</button>
  </form>

  <div class="loading" id="loading">Loading, please wait...</div>
  <h3>Response:</h3>
  <div id="response"></div>

  <script>
    document.getElementById("test-form").addEventListener("submit", function(e) {
      e.preventDefault();
      const apiKey = document.getElementById("api-key").value.trim();
      const botName = document.getElementById("bot-name").value.trim();
      const message = document.getElementById("message").value.trim();
      const stream = document.getElementById("stream-toggle").checked;
      const responseDiv = document.getElementById("response");
      const loadingDiv = document.getElementById("loading");
      responseDiv.textContent = "";
      loadingDiv.style.display = "block";
      if (stream) {
        const url = `/stream-response?api_key=${encodeURIComponent(apiKey)}&bot_name=${encodeURIComponent(botName)}&message=${encodeURIComponent(message)}`;
        const eventSource = new EventSource(url);
        eventSource.onmessage = function(event) {
          if (event.data === "[DONE]") {
            eventSource.close();
            loadingDiv.style.display = "none";
          } else {
            responseDiv.textContent += event.data;
          }
        };
        eventSource.onerror = function(error) {
          eventSource.close();
          loadingDiv.style.display = "none";
          responseDiv.textContent += " Error retrieving response.";
        };
      } else {
        fetch("/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + apiKey
          },
          body: JSON.stringify({
            model: botName,
            messages: [{ role: "user", content: message }],
            stream: false
          })
        })
        .then(res => res.json())
        .then(data => {
          loadingDiv.style.display = "none";
          if (data && data.choices && data.choices[0]) {
            responseDiv.textContent = data.choices[0].message.content;
          } else {
            responseDiv.textContent = "No response.";
          }
        })
        .catch(err => {
          loadingDiv.style.display = "none";
          responseDiv.textContent = "Error: " + err.message;
        });
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """
    Return an HTML page for testing the API.
    """
    return HTML_TEMPLATE

# =============================================================================
# 7. Additional Endpoint: List Models (OpenAI-like structure)
# =============================================================================
@app.get("/v1/models")
async def list_models():
    """
    Returns a list of available models in an OpenAI-compatible structure.
    """
    models = [
        {'id': 'Claude-3.7-Sonnet', 'name': 'Claude-3.7-Sonnet'},
        {'id': 'Claude-3.5-Sonnet', 'object': 'model', 'created': int(time.time()), 'owned_by': 'anthropic'},
        {'id': 'o3-mini', 'object': 'model', 'created': int(time.time()), 'owned_by': 'o3-mini'},
        {'id': 'DeepSeek-R1-FW', 'object': 'model', 'created': int(time.time()), 'owned_by': 'DeepSeek-R1-FW'},
        {'id': 'GPT-4o', 'object': 'model', 'created': int(time.time()), 'owned_by': 'openai'},
        {'id': 'Gemini-2.0-Pro', 'object': 'model', 'created': int(time.time()), 'owned_by': 'anthropic'},
        {'id': 'FLUX-pro-1.1', 'object': 'model', 'created': int(time.time()), 'owned_by': 'stability'},
        {'id': 'ElevenLabs', 'object': 'model', 'created': int(time.time()), 'owned_by': 'elevenlabs'},
        {'id': 'Runway', 'object': 'model', 'created': int(time.time()), 'owned_by': 'runway'},
    ]
    
    return JSONResponse(content={'object': 'list', 'data': models})

# =============================================================================
# 8. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 83))
    uvicorn.run(app, host="0.0.0.0", port=port)