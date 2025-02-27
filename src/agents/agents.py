from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph


from agents.credit import credit_agent
from schema import AgentInfo

DEFAULT_AGENT = "credits-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {

    "credits-agent": Agent(description="A credit agent", graph=credit_agent)
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
