"""
LangGraph ReAct agent with tool calling and conversational memory.

This module demonstrates how to build a ReAct agent using LangGraph that can
use tools (weather lookup, web search) and maintain conversation context
across multiple turns. Uses a local Ollama model for inference.
"""

from typing import Annotated, List

import requests
from langchain.agents import create_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Constants
MODEL_NAME = "qwen2.5:7b"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


class State(TypedDict):
    """State container for the agent's message history."""

    messages: Annotated[List, add_messages]


def get_coordinates(city: str) -> tuple[float, float] | None:
    """
    Get latitude and longitude for a city using Open-Meteo geocoding API.

    Args:
        city: Name of the city to geocode.

    Returns:
        Tuple of (latitude, longitude) if found, None otherwise.
    """
    url = f"{GEOCODING_API_URL}?name={city}&count=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None


def get_weather_info(city: str) -> str:
    """
    Retrieve the current weather information for a given city.

    Uses the Open-Meteo API which is free and requires no API key.

    Args:
        city: Name of the city to get weather for.

    Returns:
        Formatted string with weather information or error message.
    """
    coords = get_coordinates(city)
    if coords is None:
        return f"Could not find coordinates for {city}"

    lat, lon = coords
    url = f"{WEATHER_API_URL}?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data["current"]
        summary = (
            f"Weather in {city}:\n"
            f"Temperature: {current['temperature_2m']}°C\n"
            f"Humidity: {current['relative_humidity_2m']}%\n"
            f"Wind Speed: {current['wind_speed_10m']} km/h\n"
        )
        return summary
    else:
        return f"Could not retrieve weather information for {city}"


def create_tools() -> list[Tool]:
    """
    Create and return the list of tools available to the agent.

    Returns:
        List of Tool objects for the agent to use.
    """
    serpapi = SerpAPIWrapper()
    search_tool = Tool(
        name="SerpAPI",
        func=serpapi.run,
        description="A search engine tool to query real-time information from the web.",
    )

    weather_tool = Tool(
        name="GetWeather",
        func=get_weather_info,
        description="A tool to fetch the weather information for a city",
    )

    return [search_tool, weather_tool]


def run_agent(query: str, state: State | None = None) -> tuple[str, State]:
    """
    Run the agent with a query and optional conversation state.

    Args:
        query: The user's question or request.
        state: Optional conversation state from previous turns.

    Returns:
        Tuple of (agent response, updated state).
    """
    if state is None:
        state = {"messages": []}

    state["messages"].append(HumanMessage(content=query))
    response = agent_executor.invoke(state)
    state = {"messages": response["messages"]}
    return response["messages"][-1].content, state


# Module-level agent setup
llm = ChatOllama(model=MODEL_NAME, temperature=0)
agent_executor = create_agent(llm, tools=create_tools())


def main() -> None:
    """Run the interactive conversation loop with the agent."""
    conversation_state: State = {"messages": []}

    print("LangGraph ReAct Agent with Weather Tool")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Question: ")
        if query.lower() == "quit":
            break

        answer, conversation_state = run_agent(query, conversation_state)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
