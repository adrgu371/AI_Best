# AI-Powered Conversational Agent with Web Search

This Python script creates an interactive conversational AI agent powered by the Ollama language model (Llama 3.1), Brave Search API, and web scraping capabilities. Designed to provide detailed, context-aware responses (minimum 200 words), caches results for efficiency, and maintains conversation history. Key features include:

- Asynchronous web searches using Brave Search API and BeautifulSoup.
- Integration with LangChain for tool-based agent functionality.
- Enhanced CLI with `prompt_toolkit` (history, key bindings).
- Support for clearing context with `/clear` and exiting with `exit`.

Licensed under the GNU General Public License v3.0, this project is free to use, modify, and distribute. Ideal for developers interested in AI, web scraping, or conversational systems. Contributions welcome!

*Current Version: 1.0 (March 2025)*  
*Author: Gulas Adrian*  


### Github
git clone https://github.com/adrgu371/AI_Best.git

cd AI_Best

## Installation

Follow these steps to set up and run the AI-powered conversational agent on your system. This guide assumes a Unix-like environment (Linux/macOS) or Windows with adjustments noted.

### Prerequisites

- **Python 3.7+**: Ensure Python 3 is installed. Check with:
  ```bash
  python3 --version
  python3 -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip3 install aiohttp beautifulsoup4 langchain-ollama cachetools prompt-toolkit langchain certifi

  ### Ollama
  curl -fsSL https://ollama.ai/install.sh | sh  # Linux/macOS
  ollama pull llama3.1:8b
  ollama serve

  ### Replace the placeholder BRAVE_API_KEY in the script with your key:
  export BRAVE_API_KEY="your_actual_key_here"  # Add to ~/.bashrc or equivalent for persistence

  ### Run the Script With the virtual environment active and Ollama running, execute:
  python3 AI_Best.py
  ### Or make it executable
  chmod +x AI_Best.py

### SSL Errors: If SSL issues arise, ensure certifi is installed. For secure connections, update fetch_webpage_content_async in AI_Best.py:
import certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

### Alternative Models: Change llama3.1:8b in AI_Best.py to another Ollama model.
For help, file an issue at https://github.com/adrgu371/AI_Best/issues!

