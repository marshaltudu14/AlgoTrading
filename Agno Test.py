from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    model=Ollama(id="deepseek-r1:1.5b"),
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    show_tool_calls=True,
    markdown=True
)
agent.print_response("Tell me about a breaking news story from New York.", stream=True)