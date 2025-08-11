<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Pre-Work](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150?source=copy_link#247cd547af3d80709683ff380f4cba62)| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150) | [Recording!](https://us02web.zoom.us/rec/share/1YepNUK3kqQnYLY8InMfHv84JeiOMyjMRWOZQ9jfjY86dDPvHMhyoz5Zo04w_tn-.91KwoSPyP6K6u0DC)  (@@5J6DVQ)| [Session 14 Slides](https://www.canva.com/design/DAGvVPg7-mw/IRwoSgDXPEqU-PKeIw8zLg/edit?utm_content=DAGvVPg7-mw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: Production Agents](https://forms.gle/nZ7ugE4W9VsC1zXE8) | [AIE7 Feedback 8/7](https://forms.gle/juo8SF5y5XiojFyC9)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

#### ✅ Answer:

The `chunk_overlap` parameter in `RecursiveCharacterTextSplitter` is used to specify the number of overlapping characters between consecutive text chunks when splitting documents. Its primary purpose is to ensure that important information that may span across chunk boundaries is preserved, which can improve the quality of retrieval-augmented generation (RAG) by maintaining context continuity.

Trade-offs of adjusting the `chunk_overlap` value include:

- Increasing `chunk_overlap`:
  - Pros: Better context preservation across chunks, leading to potentially more accurate and coherent retrieval results.
  - Cons: Larger overlap results in more redundant data, increased storage requirements, and potentially slower processing.

- Decreasing `chunk_overlap`:
  - Pros: Reduces redundancy and storage needs, and can improve processing speed.
  - Cons: Higher risk of losing context at chunk boundaries, which may negatively impact the quality of retrieval and subsequent generation.

Choosing the optimal `chunk_overlap` depends on the specific use case, balancing the need for context preservation against efficiency considerations.

#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

#### ✅ Answer:

**Effect of k on RAGAS Metrics:**

k controls how many top documents the retriever returns.

- Increase k → Higher Context Recall, lower chance of missing relevant info, but Lower Context Precision due to more irrelevant chunks.
- Decrease k → Higher Context Precision, fewer irrelevant chunks, but Lower Context Recall from potentially missing some relevant info.

Trade-off: Larger k improves completeness; smaller k improves relevance. Choose based on whether missing facts or including noise is more harmful to your application.


#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

#### ✅ Answer:

#### Comparison of `agent` vs. `agent_helpful`

##### `agent` (Graph: `simple_agent`)
- **Flow:** `agent → (if tool_calls) action → agent → … else END`
- **Termination:** Ends immediately when the last AI message has **no `tool_calls`**.

##### `agent_helpful` (Graph: `agent_with_helpfulness`)
- **Flow:** `agent → (if tool_calls) action → agent → … else helpfulness → decision`
- When there are **no `tool_calls`**, routes to a **helpfulness evaluator** instead of ending.

---

#### Helpfulness Evaluator

- **Position in Graph:** Runs **after** the agent responds and no tools are requested.
- **Role:** Compares the **initial query** and **latest response**, producing:
  - `HELPFULNESS:Y` → Answer is satisfactory
  - `HELPFULNESS:N` → Answer is unsatisfactory
  - `HELPFULNESS:END` → Loop-limit reached

---

#### Routing Conditions

- **If `HELPFULNESS:Y`** → Route to **`end`** (terminate).
- **If `HELPFULNESS:N`** → Route **back to `agent`** for another attempt.
- **If `HELPFULNESS:END`** → Terminate due to exceeding the retry loop limit (`len(messages) > 10`).

---

## Assistant Bindings
- **`agent`** → Uses `simple_agent` graph.
- **`agent_helpful`** → Uses `agent_with_helpfulness` graph.
