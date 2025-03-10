# OpenAI-Compatible API Server using Poe API

This repository contains a FastAPI application that serves as an OpenAI-compatible API server by proxying requests to the Poe API via the [fastapi_poe](https://github.com/your-org/fastapi_poe) library. It supports both streaming (SSE) and complete JSON responses and includes a simple HTML-based test interface.

## Features

- **OpenAI API Compatible Endpoints:** Accepts chat completion requests in OpenAI's format.
- **Streaming Support:** Provides server-sent events (SSE) for streaming responses.
- **HTML Test Interface:** A built-in front-end page to test the API.
- **Model Listing:** An endpoint (`/v1/models`) that lists available models in an OpenAI-like format.
- **CORS Enabled:** Allow requests from any origins.

## Prerequisites

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [uvicorn](https://www.uvicorn.org/)
- [fastapi_poe](https://github.com/your-org/fastapi_poe) (ensure this package is installed or available)

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-github-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Create a Virtual Environment (Optional but Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install fastapi uvicorn fastapi_poe pydantic
    ```

## Configuration

- **Poe API Key:**  
  The application uses a default Poe API key by reading the `POE_API_KEY` environment variable.  
  To set it, run:
  
  ```bash
  export POE_API_KEY="your_default_poe_api_key"
