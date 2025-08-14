"""Simple LangGraph agent that can call the A2A server through the A2A protocol.

This demonstrates Activity #1: creating a LangGraph agent that can make API calls
to the A2A server through the A2A protocol.
"""
import asyncio
import logging
from typing import Dict, Any, Annotated, TypedDict, List
from uuid import uuid4

import httpx
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for the A2A client agent."""
    messages: Annotated[List, add_messages]


# Global A2A client instance
_a2a_client_instance = None

async def _get_a2a_client(base_url: str = "http://localhost:10000") -> A2AClient:
    """Get or create the A2A client."""
    global _a2a_client_instance
    if _a2a_client_instance is None:
        # Create a persistent httpx client
        httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        
        # Get the agent card
        agent_card = await resolver.get_agent_card()
        logger.info(f"Retrieved agent card: {agent_card.name}")
        
        # Create the A2A client
        _a2a_client_instance = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
    
    return _a2a_client_instance

@tool
async def call_a2a_agent(query: str) -> str:
    """Make a call to the A2A server to get information or perform tasks.
    
    Args:
        query: The question or request to send to the A2A agent
        
    Returns:
        The response from the A2A agent
    """
    try:
        logger.info(f"🔧 A2A Tool called with query: {query}")
        a2a_client = await _get_a2a_client()
        
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
        
        logger.info(f"📤 Sending query to A2A server: {query[:100]}...")
        response = await a2a_client.send_message(request)
        logger.info(f"📥 Received response from A2A server")
        
        # Extract the response content from the A2A response
        result = response.root.result
        logger.info(f"🔍 Response result type: {type(result)}")
        logger.info(f"🔍 Response result has artifacts: {hasattr(result, 'artifacts')}")
        logger.info(f"🔍 Response result has history: {hasattr(result, 'history')}")
        
        # First try to get content from artifacts (final result)
        if hasattr(result, 'artifacts') and result.artifacts:
            logger.info(f"🔍 Found {len(result.artifacts)} artifacts")
            for i, artifact in enumerate(result.artifacts):
                logger.info(f"🔍 Artifact {i}: name={artifact.name}, parts={len(artifact.parts) if artifact.parts else 0}")
                if artifact.name == 'result' and artifact.parts:
                    part = artifact.parts[0]
                    if hasattr(part, 'text'):
                        content = part.text
                    else:
                        content = str(part)
                    logger.info(f"✅ Extracted content from artifacts: {content[:100]}...")
                    return content
        
        # Fallback to history if no artifacts
        if hasattr(result, 'history') and result.history:
            logger.info(f"🔍 Found {len(result.history)} history messages")
            # Get the last assistant message from the history
            for i, message in enumerate(reversed(result.history)):
                logger.info(f"🔍 History message {i}: role={message.role}, parts={len(message.parts) if message.parts else 0}")
                if message.role == 'assistant':
                    if message.parts:
                        content = message.parts[0].text if hasattr(message.parts[0], 'text') else str(message.parts[0])
                        logger.info(f"✅ Extracted content from history: {content[:100]}...")
                        return content
        
        logger.warning("⚠️ Could not extract content from response")
        return "Received response from A2A server but couldn't extract content"
        
    except Exception as e:
        logger.error(f"Error calling A2A server: {e}")
        return f"Error calling A2A server: {str(e)}"


def create_agent_with_a2a_tool() -> ChatOpenAI:
    """Create a LangChain agent with A2A tool capabilities."""
    # Create the A2A tool
    a2a_tool = A2AClientTool()
    
    # Create the model with tools
    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools([a2a_tool.call_a2a_agent])
    
    return model_with_tools


async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that processes messages and calls the A2A agent when needed."""
    logger.info("🤖 Agent node called")
    
    # Create the model with tools
    model = ChatOpenAI(model="gpt-4o-mini")
    model_with_tools = model.bind_tools([call_a2a_agent])
    
    messages = state["messages"]
    logger.info(f"📝 Processing messages: {len(messages)} messages")
    
    # Invoke the model with tools
    response = model_with_tools.invoke(messages)
    logger.info(f"🤖 Model response type: {type(response)}")
    logger.info(f"🤖 Model response has tool_calls: {hasattr(response, 'tool_calls')}")
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"🔧 Tool calls found: {len(response.tool_calls)}")
        
        # Execute tool calls
        for i, tool_call in enumerate(response.tool_calls):
            logger.info(f"🔧 Executing tool call {i}: {tool_call['name']}")
            
            if tool_call['name'] == 'call_a2a_agent':
                query = tool_call['args']['query']
                logger.info(f"🔧 Calling A2A agent with query: {query}")
                
                try:
                    # Execute the A2A tool
                    result = await call_a2a_agent.ainvoke({"query": query})
                    logger.info(f"✅ A2A tool result: {result[:100]}...")
                    
                    # Create a tool result message
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=result,
                        tool_call_id=tool_call['id']
                    )
                    
                    return {"messages": [response, tool_message]}
                    
                except Exception as e:
                    logger.error(f"❌ Error executing A2A tool: {e}")
                    error_message = f"Error calling A2A agent: {str(e)}"
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=error_message,
                        tool_call_id=tool_call['id']
                    )
                    return {"messages": [response, tool_message]}
    
    return {"messages": [response]}


def build_simple_a2a_graph() -> StateGraph:
    """Build a simple LangGraph that can call the A2A server."""
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add the agent node - wrap the async function
    async def wrapped_agent_node(state):
        return await agent_node(state)
    
    workflow.add_node("agent", wrapped_agent_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Set end point
    workflow.add_edge("agent", END)
    
    return workflow.compile()


async def test_a2a_client_agent():
    """Test the A2A client agent."""
    print("🤖 Creating A2A Client Agent...")
    
    # Build the graph
    graph = build_simple_a2a_graph()
    
    # Test queries
    test_queries = [
        "What are the latest developments in artificial intelligence?",
        "Find me recent papers on transformer architectures",
        "What do the policy documents say about student loans?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test {i}: {query}")
        print("-" * 50)
        
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }
        
        try:
            result = await graph.ainvoke(initial_state)
            response = result["messages"][-1]
            print(f"🤖 Agent Response: {response.content}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(test_a2a_client_agent()) 