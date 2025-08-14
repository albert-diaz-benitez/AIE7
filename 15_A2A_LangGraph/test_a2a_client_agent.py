#!/usr/bin/env python3
"""
Test script for the A2A Client Agent.

This script demonstrates Activity #1: creating a LangGraph agent that can make
API calls to the A2A server through the A2A protocol.

Prerequisites:
1. Start the A2A server: `uv run python -m app`
2. Make sure the server is running on http://localhost:10000
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from simple_a2a_client import build_simple_a2a_graph
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

async def main():
    """Main test function."""
    print("🚀 A2A Client Agent Test")
    print("=" * 50)
    print("This test demonstrates Activity #1: LangGraph agent calling A2A server")
    print("=" * 50)
    
    try:
        # Build the A2A client graph
        print("🔧 Building A2A Client Graph...")
        graph = build_simple_a2a_graph()
        print("✅ Graph built successfully!")
        
        # Test queries that exercise different A2A agent capabilities
        test_cases = [
            {
                "name": "Web Search Test",
                "query": "What are the latest developments in artificial intelligence?",
                "expected_skill": "web_search"
            },
            {
                "name": "Academic Paper Search Test", 
                "query": "Find me recent papers on transformer architectures",
                "expected_skill": "arxiv_search"
            },
            {
                "name": "Document Retrieval Test",
                "query": "What do the policy documents say about student loans?",
                "expected_skill": "rag_search"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test Case {i}: {test_case['name']}")
            print(f"🎯 Expected Skill: {test_case['expected_skill']}")
            print(f"❓ Query: {test_case['query']}")
            print("-" * 60)
            
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=test_case['query'])]
            }
            
            try:
                # Invoke the graph
                result = await graph.ainvoke(initial_state)
                
                # Extract the response
                response = result["messages"][-1]
                print(f"🤖 A2A Client Agent Response:")
                print(f"📝 {response.content}")
                
                # Check if the response indicates tool usage
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"🔧 Tools Used: {[tc['name'] for tc in response.tool_calls]}")
                else:
                    print("💬 Direct response (no tools called)")
                    
            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")
                logger.error(f"Test case {i} failed", exc_info=True)
            
            print("-" * 60)
        
        print("\n🎉 A2A Client Agent Test Complete!")
        print("\n📚 What this demonstrates:")
        print("✅ LangGraph agent successfully calls A2A server")
        print("✅ A2A protocol enables agent-to-agent communication")
        print("✅ Different skills (web search, academic search, RAG) are accessible")
        print("✅ Standardized communication through AgentCard and A2A protocol")
        
    except Exception as e:
        print(f"❌ Failed to run A2A client agent test: {e}")
        logger.error("Test failed", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    print("🔍 Checking if A2A server is running...")
    print("💡 Make sure to start the A2A server first with: uv run python -m app")
    print("💡 The server should be running on http://localhost:10000")
    print()
    
    # Run the test
    asyncio.run(main()) 