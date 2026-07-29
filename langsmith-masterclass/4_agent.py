from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

import os
import requests

load_dotenv()

# Load API Key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Search Tool
search_tool = DuckDuckGoSearchRun()


# Weather Tool
@tool
def get_weather_data(city: str) -> dict:
    """
    Fetch the current weather data for a given city using OpenWeatherMap API.
    """

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)
    return response.json()


# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Create Agent
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt=(
        "You are a helpful AI assistant. "
        "Use the available tools whenever necessary to answer the user's questions."
    ),
)

# Invoke
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Identify the bith place of kalpana chawla and give its current temperature",
            }
        ]
    }
)

print(response)
print(response["messages"][-1].content)