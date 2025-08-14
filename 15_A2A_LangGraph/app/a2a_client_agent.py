"""LangGraph agent that can call the A2A server through the A2A protocol.

This agent demonstrates how to create a LangGraph graph that can interact with
other agents through the A2A protocol, enabling agent-to-agent communication.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    """State schema for A2A client agent graphs."""
    messages: Annotated[List, add_messages]
    a2a_response: str | None


class A2ACallTool(BaseTool):
    """Tool for making calls to the A2A server."""
    
    name: str = "a2a_call"
    description: str = "Make a call to the A2A server to get information or perform tasks"
    
    def __init__(self, a2a_client: A2AClient):
        super().__init__()
        self.a2a_client = a2a_client
    
    async def _arun(self, query: str) -> str:
        """Make an async call to the A2A server."""
        try:
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            response = await self.a2a_client.send_message(request)
            
            # Extract the response content from the A2A response
            if hasattr(response.root, 'result') and hasattr(response.root.result, 'history'):
                # Get the last assistant message from the history
                for message in reversed(response.root.result.history):
                    if message.role == 'assistant':
                        return message.parts[0].text if message.parts else "No response content"
            
            return "Received response from A2A server but couldn't extract content"
            
        except Exception as e:
            logging.error(f"Error calling A2A server: {e}")
            return f"Error calling A2A server: {str(e)}"


async def create_a2a_client(base_url: str = "http://localhost:10000") -> A2AClient:
    """Create and configure an A2A client."""
    httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
    
    # Get the agent card
    agent_card = await resolver.get_agent_card()
    
    # Create the A2A client
    return A2AClient(httpx_client=httpx_client, agent_card=agent_card)


async def call_a2a_agent(state: Dict[str, Any], a2a_client: A2AClient) -> Dict[str, Any]:
    """Call the A2A agent and return the response."""
    # Get the last user message
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        query = last_message.content
        
        # Create the A2A call tool
        a2a_tool = A2ACallTool(a2a_client)
        
        # Make the call
        response = await a2a_tool._arun(query)
        
        return {
            "messages": [AIMessage(content=response)],
            "a2a_response": response
        }
    
    return {"messages": [AIMessage(content="No valid query found")]}


def build_a2a_client_graph(
    model: ChatOpenAI,
    a2a_client: A2AClient,
    system_instruction: str = "You are a helpful assistant that can call other agents through the A2A protocol."
) -> StateGraph:
    """Build a LangGraph that can call the A2A server."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes - wrap the async function
    async def a2a_call_node(state):
        return await call_a2a_agent(state, a2a_client)
    
    workflow.add_node("a2a_call", a2a_call_node)
    
    # Set entry point
    workflow.set_entry_point("a2a_call")
    
    # Set end point
    workflow.add_edge("a2a_call", END)
    
    return workflow.compile()


async def main():
    """Example usage of the A2A client agent."""
    # Create the A2A client
    a2a_client = await create_a2a_client()
    
    # Create the LangGraph
    model = ChatOpenAI(model="gpt-4o-mini")
    graph = build_a2a_client_graph(model, a2a_client)
    
    # Test the graph
    initial_state = {
        "messages": [
            HumanMessage(content="What are the latest developments in artificial intelligence?")
        ],
        "a2a_response": None
    }
    
    result = await graph.ainvoke(initial_state)
    print("A2A Client Agent Response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 