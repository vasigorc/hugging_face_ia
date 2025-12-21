# Chapter 9. Programming agents

This chapter introduces the concept of programming agents, focusing on how AI agents can understand, reason, act and respond
to real-world tasks. The author highlights that while specialized models are effective for well-defined tasks, managing
ambiguous, multi step, or dynamic tasks becomes challenging. Agents address this by leveraging Large Language Models (LLMs)
to reason, plan, and delegate tasks, breaking down complex problems into smaller subtasks and using appropriate tools or
models. Agents are able to simplify complex workflows. In this chapter the author sets the stage for building intelligent
agents that can manage state, memory, and tool selection in multi step reasoning pipelines.

## Using LangGraph with tools and memory

The example chosen for this chapter highlights LangGraph, a LangChain extension recommended for building agents with tools
and memory. LangGraph provides flexible graph-based workflow that allows for complex, stateful, and multi step logic. It enables
easy integration of language models, tools, and external APIs, supports branching and dynamic flow control, and is especially
valuable when your application requires persistent memory across multiple steps. This makes it ideal for advanced agents that
need to reason, use tools, and maintain conversational context.

### Example: Weather Agent with Tool Calling

The `langgraph_example.py` script demonstrates a ReAct agent that uses a weather tool to fetch real-time weather data.
The agent can reason about when to use tools and maintain conversation context across multiple turns:

```
Question: What is the weather like in Sainte-Sophie, QC?
Answer: I encountered an issue while trying to fetch the weather information for Sainte-Sophie, QC.
It seems that there might be a problem with finding its geographical coordinates.
Could you please provide more details or try another nearby city?

Question: Sure, what about Saint-Jerome, QC?
Answer: The current weather conditions in Saint-Jérôme, QC are mostly cloudy. Here's a breakdown of the details:
- Temperature: 20°F
- Precipitation: 10%
- Humidity: 66%
- Wind: 11 mph

Question: Can you recommend me some activities in this city for this weather?
Answer: Given the mostly cloudy conditions and a temperature of 20°F, here are some activity recommendations
for Saint-Jérôme, QC:

1. **Indoor Activities:**
   - Visit local museums or art galleries.
   - Go to the cinema or watch a movie at home with friends.
   - Explore indoor shopping centers or malls.

2. **Outdoor Activities (if you prefer cooler weather):**
   - Take a walk in a nearby park, such as Parc de la Chute-Montmorency.
   - Enjoy a leisurely bike ride on one of the city's trails.
   - Go skating at a local rink if there is ice available.

3. **Winter Sports:**
   - If you're up for it, hit the slopes at one of the nearby ski resorts like Mont-Tremblant or Bromont.
```

This example shows the agent:
- Using the `GetWeather` tool to fetch real-time weather data via the Open-Meteo API
- Gracefully handling errors when coordinates cannot be found
- Maintaining conversational context to understand "this city" refers to Saint-Jérôme from the previous turn
- Combining tool results with its own knowledge to provide relevant activity recommendations
