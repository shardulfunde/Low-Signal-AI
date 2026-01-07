# Low-Signal-AI

**Low-Signal-AI** is a FastAPI-based backend application designed to generate educational content and provide AI chat capabilities. It leverages LangChain and Cerebras (high-performance LLM inference) to create personalized learning paths, generate quizzes, and facilitate streaming chat interactions.

Key features include support for multiple languages (**English, Hindi, Marathi**) and age-appropriate content generation.

## 🚀 Features

* **AI Chatbot:** A streaming chat interface powered by Llama-3.3-70b.
* **Test Generator:** Automatically creates Multiple Choice Question (MCQ) tests based on topic and difficulty.
* **Learning Path Generator:**
    * **Topic Planner:** Breaks down a subject into sequential topics based on learner age.
    * **Topic Expander:** Detailed explanations and practice questions for specific topics.
* **Multi-Language Support:** Content generation in English (`en`), Hindi (`hi`), and Marathi (`mr`).

## 🛠️ Tech Stack

* **Framework:** FastAPI
* **Server:** Uvicorn
* **AI/LLM Orchestration:** LangChain
* **Inference Provider:** Cerebras (using `langchain-google-genai` and `langchain-cerebras`)
* **Models Used:**
    * `gemini-2.5-flash-lite`

## 📋 Prerequisites

* Python 3.9+
* A Google Gemini API Key (and potentially Cerebras API Key if you uncomment specific imports).

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Low-Signal-AI
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration:**
    Create a `.env` file in the root directory. You must add your API keys here:
    ```env
    CEREBRAS_API_KEY=your_cerebras_api_key_here
    # GOOGLE_API_KEY=your_google_key (if required by future updates)
    ```

## 🏃‍♂️ Running the Application

Start the server using Uvicorn:

```bash
uvicorn main:app --reload

