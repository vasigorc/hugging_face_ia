from typing import Annotated, List
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SerpAPIWrapper
import requests

serpapi = SerpAPIWrapper()
search_tool = Tool(
    name="SepAPI",
    func=serpapi.run,
    description="A search engine tool to query real-time information from the web.",
)

def get_coordinates(city: str) -> tuple[float, float] | None:
    """Get latitude and longitude for a city using Open-Meteo geocoding API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and data["results"]:
            return data["results"][0]["latitude"], data["results"][0]["longitude"]
    return None


def get_weather_info(city: str) -> str:
    """
    Retrieve the current weather information for a given city using Open-Meteo (free, no API key).
    """
    coords = get_coordinates(city)
    if coords is None:
        return f"Could not find coordinates for {city}"

    lat, lon = coords
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
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


weather_tool = Tool(
    name="GetWeather",
    func=get_weather_info,
    description="A tool to fetch the weather information for a city",
)

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
# Create a ReAct agent
agent_executor = create_agent(llm, tools=[search_tool, weather_tool])


class State(TypedDict):
    messages: Annotated[List, add_messages]


def run_agent(query: str, state: State | None = None) -> tuple[str, State]:
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    print(response)
    if state is None:
        state = {"messages": []}

    state["messages"].append(("user", query))
    response = agent_executor.invoke(state)
    state = {"messages": response["messages"]}
    return response["messages"][-1].content, state
