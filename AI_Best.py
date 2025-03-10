#!/usr/bin/env python3
# Copyright (C) 2025 Gulas Adrian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import aiohttp
import asyncio
import warnings
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain.agents import Tool, AgentType, initialize_agent
import ssl
from cachetools import TTLCache
import os
import json
from datetime import datetime
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# API Keys
BRAVE_API_KEY = ""

# Cache for internet data (1 hour TTL, 100 entries)
cache = TTLCache(maxsize=100, ttl=3600)

# Conversation history file
CONVERSATION_FILE = "conversation_history.json"

# Load/save conversation history
def load_conversation():
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r") as file:
            return json.load(file)
    return []

def save_conversation(conversation):
    with open(CONVERSATION_FILE, "w") as file:
        json.dump(conversation, file, indent=4)

conversation_history = load_conversation()

# Tools
async def search_brave_async(query):
    if query in cache:
        return cache[query]
    async with aiohttp.ClientSession() as session:
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": BRAVE_API_KEY}
            params = {"q": query + " after:2023", "count": 5}
            ssl_context = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None)
            async with session.get(url, headers=headers, params=params, ssl=ssl_context) as response:
                data = await response.json()
                results = data.get("web", {}).get("results", [])
                if results:
                    combined_content = ""
                    for result in results[:3]:
                        url = result.get("url")
                        if url:
                            content = await fetch_webpage_content_async(url)
                            if "Error" not in content:
                                combined_content += f"Source: {url}\nContent: {content}\n\n"
                    cache[query] = combined_content if combined_content else "No useful post-2023 information found."
                else:
                    cache[query] = "No useful post-2023 information found."
                return cache[query]
        except Exception as e:
            cache[query] = f"Error with Brave Search API: {str(e)}"
            return cache[query]

async def fetch_webpage_content_async(url):
    if url in cache:
        return cache[url]
    async with aiohttp.ClientSession() as session:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            async with session.get(url, headers=headers, ssl=ssl_context) as response:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                content = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
                result = content[:1000] if content else "No readable content found."
                cache[url] = result
                return result
        except Exception as e:
            cache[url] = f"Error fetching webpage: {str(e)}"
            return cache[url]

tools = [
    Tool(name="brave_search", func=lambda q: asyncio.run(search_brave_async(q)), description="Search the web using Brave Search API for events after 2023."),
    Tool(name="fetch_webpage_content", func=lambda u: asyncio.run(fetch_webpage_content_async(u)), description="Fetch text from a webpage."),
]

# Initialize LLM
llm = OllamaLLM(model="llama3.1:8b", temperature=0.7, top_p=0.9, max_tokens=1500)  # Increased max_tokens for longer responses

# Current date and time dynamically
CURRENT_DATE = datetime.now().strftime("%B %d, %Y at %I:%M %p PDT")

# Decision function
def should_search_internet(query, history):
    if "latest" in query.lower() or any(keyword in query.lower() for keyword in ["news", "2024", "2025", "polls", "search"]):
        return True
    if history and "is not" in query.lower() and "is" in query.lower():
        return True  # Trigger search on correction
    # Add specific name-based triggers
    if any(name.lower() in query.lower() for name in ["elon musk", "calin georgescu", "musk", "georgescu"]):
        return True
    return False

# Response generation
def generate_response(query, external_data=None, use_history=False):
    static_info = "No specific static knowledge available up to December 2023."
    if use_history and conversation_history:
        context = "\n".join(f"User: {entry['user']}\nAI: {entry['ai']}" for entry in conversation_history[-2:])
        prompt = f"Current Date: {CURRENT_DATE}\nConversation History:\n{context}\n\nStatic Knowledge (up to Dec 2023): {static_info}\nExternal Data (post-2023): {external_data or 'None'}\nQuestion: {query}\nGenerate a detailed, comprehensive response as a news summary with numbered points if 'latest', 'news', or 'search' is in the query, synthesizing static knowledge and external data into a cohesive update up to the current date. Ensure responses are at least 200 words long, providing thorough analysis and context. If the user corrects a name (e.g., 'is not X, is Y'), adjust to the corrected entity and apologize for the mix-up. Always provide a response, noting gaps if data is limited. Avoid speculation and do not mention knowledge cutoff dates:"
    else:
        prompt = f"Current Date: {CURRENT_DATE}\nStatic Knowledge (up to Dec 2023): {static_info}\nExternal Data (post-2023): {external_data or 'None'}\nQuestion: {query}\nGenerate a detailed, comprehensive response as a news summary with numbered points if 'latest', 'news', or 'search' is in the query, synthesizing static knowledge and external data into a cohesive update up to the current date. Ensure responses are at least 200 words long, providing thorough analysis and context. If the user corrects a name (e.g., 'is not X, is Y'), adjust to the corrected entity and apologize for the mix-up. Always provide a response, noting gaps if data is limited. Avoid speculation and do not mention knowledge cutoff dates:"
    response = llm.invoke(prompt).replace(prompt, "").strip()
    return response

# Main async invocation
async def main_async(query, use_history=True):
    global conversation_history

    # Dynamic entity detection with correction
    if "is not" in query.lower() and "is" in query.lower():
        parts = query.lower().split("is")
        wrong_name = parts[1].split(",")[0].strip()
        correct_name = parts[-1].strip()
        query = f"Sorry for the mix-up! {query.replace(wrong_name, correct_name).replace('is not', '').replace(', is', '')}"

    # Search if needed
    if should_search_internet(query, conversation_history if use_history else []):
        search_results = await search_brave_async(query)
        if isinstance(search_results, list) and search_results:
            combined_content = ""
            for result in search_results[:3]:
                url = result.get("url")
                if url:
                    content = await fetch_webpage_content_async(url)
                    if "Error" not in content:
                        combined_content += f"Source: {url}\nContent: {content}\n\n"
            external_data = combined_content if combined_content else "Limited post-2023 search data available."
        else:
            external_data = search_results
    else:
        external_data = None

    # Generate response
    response = generate_response(query, external_data, use_history)
    conversation_history.append({"user": query, "ai": response, "timestamp": datetime.now().isoformat()})
    save_conversation(conversation_history)
    return response

def agent_invoke_async(query, use_history=True):
    return asyncio.run(main_async(query, use_history))

# Initialize agent
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False, handle_parsing_errors=True)

# Enhanced input with arrow keys, backspace, and history
bindings = KeyBindings()
session = PromptSession(
    "Your question (type 'exit' to quit or '/clear' to ignore prior context): ",
    history=FileHistory("prompt_history.txt"),
    key_bindings=bindings,
    multiline=False,
)

# Test loop with /clear command
print("Ask a question (type 'exit' to quit or '/clear' to ignore prior context):")
use_history = True
while True:
    query = session.prompt()
    if query.lower() == "exit":
        break
    elif query.lower() == "/clear":
        use_history = False
        print("Alright, let’s switch gears! What would you like to talk about next?\n")
        continue
    try:
        # Use history unless /clear was just entered
        response = agent_invoke_async(query, use_history)
        print(f"Answer: {response}\n")
        # Reset use_history to True after one query without history
        use_history = True
    except Exception as e:
        print(f"Error: {str(e)}\n")
