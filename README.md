# AI LLM Agents (Artificial Intelligence - Large Language Models)

- [SAMPLE CODE](#sample-code)
- [AGENTS (generated code)](#agents-generated-code)
- [RAG (generated code)](#rag-generated-code)
- [AI LLM Agents](#ai-llm-agents)
- [Gradio UI](#gradio-ui)
- [Streamlit UI](#streamlit-ui)
- [Streamlit UI (2)](#streamlit-ui-2)
- [30 Days of Streamlit](#30-days-of-streamlit)
- [GUIDE (generated code)](#0-guide-generated-by-claude-llm)
- [PROJECTS (generated code)](#0-projects-generated-by-claude-llm)

# SAMPLE CODE

- **AG2 (formerly AutoGen)**
  - **Quick Start** [101-ag2-quick-start.ipynb](100-ag2/101-ag2-quick-start.ipynb)✅
  - **Basic Concepts**
    - **Structured outputs** [102-ag2-basics-1.ipynb](100-ag2/102-ag2-basics-1.ipynb)✅
    - **ConversableAgent, Human in the loop** [103-ag2-basics-2.ipynb](100-ag2/103-ag2-basics-2.ipynb)✅
    - **Orchestrating agents**
      - **Sequential chat** [104-ag2-basics-3.ipynb](100-ag2/104-ag2-basics-3.ipynb)✅
      - **Nested chat** [105-ag2-basics-4.ipynb](100-ag2/105-ag2-basics-4.ipynb)✅
      - **Swarm** [106-ag2-basics-5.ipynb](100-ag2/106-ag2-basics-5.ipynb)✅
      - **Swarm Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm/deep-dive)⚠️
      - **Ending a chat** [107-ag2-basics-6.ipynb](100-ag2/107-ag2-basics-6.ipynb)✅ TRY⚠️
    - **Tools**
      - **Tools Basics** [108-ag2-basics-7.ipynb](100-ag2/108-ag2-basics-7.ipynb)✅
      - **Tools with Secrets** [109-ag2-basics-8.ipynb](100-ag2/109-ag2-basics-8.ipynb)✅
      - **Tools with Dependency Injection** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/tools_dependency_injection)⚠️
      - **Interoperability**
        - **LangChain Tools Integration** [110-ag2-basics-9.ipynb](100-ag2/110-ag2-basics-9.ipynb)✅
        - **CrewAI Tools Integration** [111-ag2-basics-10.ipynb](100-ag2/111-ag2-basics-10.ipynb)✅
        - **PydanticAI Tools Integration** [112-ag2-basics-11.ipynb](100-ag2/112-ag2-basics-11.ipynb)✅
  - **Advanced Concepts**
    - **RAG** [113-ag2-advanced-1.ipynb](100-ag2/113-ag2-advanced-1.ipynb)✅
      - **Using Neo4j’s graph database with AG2 agents for Q&A notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_graph_rag_neo4j)⚠️
      - **Trip planning with a FalkorDB GraphRAG agent using a Swarm notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_graphrag_trip_planner)⚠️
    - **GroupChat**
      - **Overview** [114-ag2-advanced-2.ipynb](100-ag2/114-ag2-advanced-2.ipynb)✅
      - **Group Chat with Tools** [115-ag2-advanced-3.ipynb](100-ag2/115-ag2-advanced-3.ipynb)✅
      - **Customized GroupChat flows** [116-ag2-advanced-4.ipynb](100-ag2/116-ag2-advanced-4.ipynb)✅
        - **More GroupChat examples**
          - **GroupChat with Customized Speaker Selection Method** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_customized)⚠️
          - **GroupChat with Coder and Visualization Critic** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_vis)⚠️
          - **GroupChat with Retrieval-Augmented Generation** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_groupchat_RAG)⚠️
          - **Implementing Swarm with a GroupChat** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_w_groupchat_legacy)⚠️
      - **Resuming a Group Chat** [117-ag2-advanced-5.ipynb](100-ag2/117-ag2-advanced-5.ipynb)✅ TRY⚠️
    - **Swarm**
      - **Deep-dive** [118-ag2-advanced-6.ipynb](100-ag2/118-ag2-advanced-6.ipynb)✅ TRY⚠️
      - **Nested chats** [119-ag2-advanced-7.ipynb](100-ag2/119-ag2-advanced-7.ipynb)✅ TRY⚠️
      - **Concepts Code** [120-ag2-advanced-8.ipynb](100-ag2/120-ag2-advanced-8.ipynb)✅
      - **Use Case example** [121-ag2-advanced-9.ipynb](100-ag2/121-ag2-advanced-9.ipynb)✅✅✅
      - **More Swarm examples**
        - **Introduction to Swarm notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm)⚠️
        - **Swarm with GraphRAG notebook** [TODO](https://docs.ag2.ai/docs/use-cases/notebooks/notebooks/agentchat_swarm_graphrag_trip_planner)⚠️
    - **RealtimeAgent** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/realtime-agent/index)⚠️
    - **Code Execution** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/code-execution)⚠️
    - **Conversation Patterns Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/conversation-patterns-deep-dive)⚠️
    - **LLM Config Deep-dive** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/llm-configuration-deep-dive)⚠️
    - **Pattern Cookbook** [TODO](https://docs.ag2.ai/docs/user-guide/advanced-concepts/pattern-cookbook/overview)⚠️
  - **Model Providers** [TODO](https://docs.ag2.ai/docs/user-guide/models)⚠️
  - **Reference Agents** [TODO](https://docs.ag2.ai/docs/user-guide/reference-agents/index)⚠️
  - **Reference Tools** [TODO](https://docs.ag2.ai/docs/user-guide/reference-tools/index)⚠️
- **CrewAI**
  - **Quick Start** [201_crewai_quickstart.ipynb](200-crewai/201_crewai_quickstart.ipynb)✅
- **LangGraph**
  - **Quick Start** [301_langgraph_quickstart.ipynb](300-langgraph/301_langgraph_quickstart.ipynb)✅
  - **Tutorials**
    - **Agentic RAG** [302_langgraph_agentic_rag.ipynb](300-langgraph/302_langgraph_agentic_rag.ipynb)✅
    - **Code Assistant** [303_langgraph_code_assistant.ipynb](300-langgraph/303_langgraph_code_assistant.ipynb)✅
- **LlamaIndex**
  - **Quick Start** [401_llamaindex_quickstart.ipynb](400-llamaindex/401_llamaindex_quickstart.ipynb)✅
  - **Starter Examples**
    - **Starter Tutorial (Using OpenAI)** [402_llamaindex_starter_openai.ipynb](400-llamaindex/402_llamaindex_starter_openai.ipynb)✅
    - **Starter Tutorial (Using Local LLMs)** [403_llamaindex_starter_local_llms.ipynb](400-llamaindex/403_llamaindex_starter_local_llms.ipynb)✅ TRY(>32GB)⚠️
  - **Examples**
    - **AgentWorkflow Basic Introduction** [404_llamaindex_agent_workflow.ipynb](400-llamaindex/404_llamaindex_agent_workflow.ipynb)✅
  - **Learn**
    - **Building agents (5 parts)** [405_llamaindex_building_agents.ipynb](400-llamaindex/405_llamaindex_building_agents.ipynb)✅
      - (tools, state, streaming, loop, multi-agent)⚠️TRY(multi-agent, result to dict)
    - **Building Workflows** [TODO](https://docs.llamaindex.ai/en/stable/understanding/workflows/)⚠️
    - **Building a RAG pipeline** [406_llamaindex_building_rag.ipynb](400-llamaindex/406_llamaindex_building_rag.ipynb)✅
    - **Structured Data Extraction** [407_llamaindex_structured_data.ipynb](400-llamaindex/407_llamaindex_structured_data.ipynb)✅
    - **Tracing and Debugging** [TODO](https://docs.llamaindex.ai/en/stable/understanding/tracing_and_debugging/tracing_and_debugging/)⚠️
    - **Evaluating** [TODO](https://docs.llamaindex.ai/en/stable/understanding/evaluating/evaluating/)⚠️
    - **Full-Stack Web Application** [TODO](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/apps/)⚠️
      - **LlamaIndex Starter Pack** [TODO](https://github.com/logan-markewich/llama_index_starter_pack)⚠️⚠️⚠️
  - **Component Guides (Agents)** [TODO](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)⚠️
    - **Examples/Module Guides** [TODO](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules/)⚠️
  - **Use Cases (Question-Answering (RAG))** [TODO](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/)⚠️
    - **RAG over Unstructured Docs (Semantic search, Summarization)** [410_llamaindex_unstructured_docs.ipynb](400-llamaindex/410_llamaindex_unstructured_docs.ipynb)✅
    - **QA over Structured Data (Text-to-SQL)** [408_llamaindex_text-to-sql.ipynb](400-llamaindex/408_llamaindex_text-to-sql.ipynb)✅
    - **QA over Structured Data (Text-to-Pandas)** [409_llamaindex_text-to-pandas.ipynb](400-llamaindex/409_llamaindex_text-to-pandas.ipynb)✅

<br>

# AGENTS (generated code)

- **001: Autogen AI Agent, with 2 different agents**
  - by **GitHub Copilot** [001-1-github-copilot.ipynb](001/001-1-github-copilot.ipynb)
  - by **DeepSeek** [001-2-deepseek.ipynb](001/001-2-deepseek.ipynb)
  - by **ChatGPT** [001-3-chatgpt.ipynb](001/001-3-chatgpt.ipynb)
  - by **Claude** [001-4-claude.ipynb](001/001-4-claude.ipynb)
  - by **Cursor** [001-5-cursor-1.ipynb](001/001-5-cursor-1.ipynb)
  - by **Cursor** [001-5-cursor-2.ipynb](001/001-5-cursor-2.ipynb)
- **002: CrewAI AI Agent, with 2 different agents**
  - by **DeepSeek** [002-2-deepseek.ipynb](002/002-2-deepseek.ipynb)
  - by **ChatGPT** [002-3-chatgpt.ipynb](002/002-3-chatgpt.ipynb)✅
  - by **Claude** [002-4-claude.ipynb](002/002-4-claude.ipynb)
  - by **Cursor** [002-5-cursor.ipynb](002/002-5-cursor.ipynb)✅

<br>

# RAG (generated code)
- **Pinecone (vector store)**
  - (quickstart) [rag-000.ipynb](rag/rag-000.ipynb)✅
  - (with LangChain) [rag-001.ipynb](rag/rag-001.ipynb)✅ + data (3 files) (TXT, MD, PDF)
  - (with LangChain) [rag-002.ipynb](rag/rag-002.ipynb)✅ + data (6 files) (MD, IPYNB, PDF, DOCX, XLSX, HTML)
    - ⚠️ IMPROVE LOAD, CHUNK, METADATA FOR ALL FILE TYPES

<br>

# AI LLM Agents

- [LLM Agents](https://www.promptingguide.ai/research/llm-agents) *by Prompt Engineering Guide*
- **AutoGen (Microsoft)**
  - [PyPI](https://pypi.org/user/AutoGenDevs/) autogen-core, autogen-agentchat, autogen-ext
  - [GitHub](https://github.com/microsoft/autogen)
  - [AutoGen](https://microsoft.github.io/autogen/stable/index.html)
  - [AgentChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)
- **AG2**
  - [PyPI](https://pypi.org/user/autogen-ai/) autogen, ag2
  - [GitHub](https://github.com/ag2ai/ag2)
  - [AG2](https://ag2.ai/)
- **CrewAI**
  - [PyPI](https://pypi.org/project/crewai/) crewai
  - [GitHub](https://github.com/crewAIInc/crewAI)
  - [CrewAI](https://www.crewai.com/)
- **LangChain**
  - [Agents](https://python.langchain.com/docs/how_to/#agents)
  - [Build an Agent](https://python.langchain.com/docs/tutorials/agents/)
  - [LangGraph](https://langchain-ai.github.io/langgraph/)
- **LlamaIndex**
  - [Agents (use cases)](https://docs.llamaindex.ai/en/stable/use_cases/agents/)
  - [Agents (module guides)](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)

<br>

# Gradio UI
- **Sample code with OpenAI** [app.py](gradio/app.py)✅

<br>

# Streamlit UI
- **AG2** (GroupChat, GroupChatManager) [ag2-groupchat.ipynb](streamlit/ag2-groupchat.ipynb)✅
- **AG2** (Swarm) [ag2-swarm-1.ipynb](streamlit/ag2-swarm-1.ipynb)✅
- **AG2** (Swarm) [ag2-swarm-2.ipynb](streamlit/ag2-swarm-2.ipynb)✅
- **AG2** (Swarm + Streamlit) [ag2-swarm-2-streamlit.py](streamlit/ag2-swarm-2-streamlit.py)✅ [Recipe Creator Swarm] [PDF](streamlit/ag2-swarm-2-streamlit.pdf) [MD](streamlit/recipe_italian.md)
- **AG2** (Swarm + Streamlit) [main.py](streamlit/main.py)✅ [AI Game Design Agent Team] [PDF](streamlit/main.pdf)
- **CrewAI** (Agents + Streamlit) [crewai-1-streamlit.py](streamlit/crewai-1-streamlit.py)✅ [Travel Planning Crew] [PDF](streamlit/crewai-1-streamlit.pdf)

<br>

# Streamlit UI (2)

- **Basic Streamlit by GitHub Copilot** (controls, data, chart, file uploader, csv) [000-streamlit-basic-copilot.py](streamlit-2/000-streamlit-basic-copilot.py)✅ [PDF](streamlit-2/streamlit-output/000%20Basic%20Streamlit%20by%20GitHub%20Copilot.pdf)
- **Re-write your text** (openai, prompt template) [001-streamlit-redaction-improver.py](streamlit-2/001-streamlit-redaction-improver.py)✅ [PDF](streamlit-2/streamlit-output/001%20Re-write%20your%20text.pdf)
- **Blog Post Generator** (openai, prompt template) [002-streamlit-blog-post-generator.py](streamlit-2/002-streamlit-blog-post-generator.py)✅ [PDF](streamlit-2/streamlit-output/002%20Blog%20Post%20Generator.pdf)
- **AI Long Text Summarizer** (openai, upload file, recursive character text splitter, summarize chain) [003-streamlit-split-and-summarize.py](streamlit-2/003-streamlit-split-and-summarize.py)✅ [PDF](streamlit-2/streamlit-output/003%20AI%20Long%20Text%20Summarizer%20(date-cz).pdf), [PDF](streamlit-2/streamlit-output/003%20AI%20Long%20Text%20Summarizer%20(david-lynch).pdf)
- **Writing Text Summarization** (openai, character text splitter, summarize chain) [004-streamlit-text-summarization.py](streamlit-2/004-streamlit-text-summarization.py)✅ [PDF](streamlit-2/streamlit-output/004%20Writing%20Text%20Summarization.pdf)
- **Extract Key Information from Product Reviews** (openai, prompt template, format output) [005-streamlit-extract-json-from-review.py](streamlit-2/005-streamlit-extract-json-from-review.py)✅ [PDF](streamlit-2/streamlit-output/005%20Extract%20Key%20Information%20from%20Product%20Reviews.pdf)
- **Evaluate a RAG App** (openai, upload file, rag with faiss, qa eval chain) [009-streamlit-evaluate-QandA-from-long-document.py](streamlit-2/009-streamlit-evaluate-QandA-from-long-document.py)✅ [PDF](streamlit-2/streamlit-output/009%20Evaluate%20a%20RAG%20App%20-%20CORRECT.pdf), [PDF](streamlit-2/streamlit-output/009%20Evaluate%20a%20RAG%20App%20-%20INCORRECT.pdf)

## 30 Days of Streamlit
- [Get started with Streamlit](https://docs.streamlit.io/get-started)<br>
  [30 Days of Streamlit](https://30days.streamlit.app/)

- **Install conda**<br>
  [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
- **Terminal**
  ```
  conda create -n stenv python=3.9
  conda activate stenv
  pip install streamlit
  streamlit hello
  ```
  ```
  conda create --name myenv
  conda create --name myenv python=3.8 numpy
  conda activate myenv
  conda install --file requirements.txt
  conda deactivate

  conda info --envs
  conda env list
  conda env remove --name myenv
  ```
  ```
  python -m venv venv
  .\venv\Scripts\activate  # Windows
  source venv/bin/activate  # macOS
  pip install -r requirements.txt
  ```
- `streamlit_app.py`<br>`streamlit_app_layout.py`<br>`streamlit_app_dashboard.py`<br>`streamlit_app_shap.py`<br>`streamlit_app_zero_shot_classifier.py`<br>`streamlit_app_art.py`
  ```
  streamlit run streamlit_app.py
  ```

## 0. GUIDE <small>(*Generated by Claude LLM*)</small>
*<small>Claude Sonnet 4, Claude 3.7 Sonnet Thinking</small>*

- **1. AI Agents Development Course Guide** 📗 [01-ai-agents-en.md](0-GUIDE/01-ai-agents-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [01-ai-agents-cz.md](0-GUIDE/01-ai-agents-cz.md)
  01. **AI API and First Agent** 📗 [01-01-cs4a.md](0-GUIDE/01-01-cs4a.md) 📗 [01-01a.md](0-GUIDE/01-01a.md)
  02. **Databases for Agents** 📗 [01-02-cs4a.md](0-GUIDE/01-02-cs4a.md) 📗 [01-02a.md](0-GUIDE/01-02a.md)
  03. **Model Context Protocol (MCP)** 📗 [01-03-cs4a.md](0-GUIDE/01-03-cs4a.md) 📗 [01-03a.md](0-GUIDE/01-03a.md)
  04. **Automation and Workflow with n8n** 📗 [01-04-cs4a.md](0-GUIDE/01-04-cs4a.md) 📗 [01-04a.md](0-GUIDE/01-04a.md)
  05. **Custom Agent Framework** 📗 [01-05-cs4a.md](0-GUIDE/01-05-cs4a.md) 📗 [01-05a.md](0-GUIDE/01-05a.md)
  06. **LangChain and LangGraph** 📗 [01-06-cs4a.md](0-GUIDE/01-06-cs4a.md) 📗 [01-06a.md](0-GUIDE/01-06a.md)
  07. **Semantic Kernel and Autogen** 📗 [01-07-cs4a.md](0-GUIDE/01-07-cs4a.md) 📗 [01-07a.md](0-GUIDE/01-07a.md)
  08. **AI Agent in Practice: OpenAI Operator Style** 📗 [01-08-cs4a.md](0-GUIDE/01-08-cs4a.md) 📗 [01-08a.md](0-GUIDE/01-08a.md)
  09. **Introduction to Reinforcement Learning** 📗 [01-09-cs4a.md](0-GUIDE/01-09-cs4a.md) 📗 [01-09a.md](0-GUIDE/01-09a.md)
  10. **RL Agent - Practical Project** 📗 [01-10-cs4a.md](0-GUIDE/01-10-cs4a.md) 📗 [01-10a.md](0-GUIDE/01-10a.md)
  11. **Summary and Discussion** 📗 [01-11-cs4a.md](0-GUIDE/01-11-cs4a.md) 📗 [01-11a.md](0-GUIDE/01-11a.md)

- **2. AI Developer Course Guide** 📗 [02-ai-developer-en.md](0-GUIDE/02-ai-developer-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [02-ai-developer-cz.md](0-GUIDE/02-ai-developer-cz.md)
  01. **Introduction to Neural Networks and Generative AI** 📗 [02-01-cs4a.md](0-GUIDE/02-01-cs4a.md) 📗 [02-01a.md](0-GUIDE/02-01a.md), [02-01b.md](0-GUIDE/02-01b.md)
  02. **Prompt Design and LLM Evaluation** 📗 [02-02-cs4a.md](0-GUIDE/02-02-cs4a.md) 📗 [02-02a.md](0-GUIDE/02-02a.md)
  03. **Training Data Preparation** 📗 [02-03-cs4a.md](0-GUIDE/02-03-cs4a.md) 📗 [02-03a.md](0-GUIDE/02-03a.md)
  04. **OpenAI Models and Fine-tuning** 📗 [02-04-cs4a.md](0-GUIDE/02-04-cs4a.md) 📗 [02-04a.md](0-GUIDE/02-04a.md)
  05. **HuggingFace Introduction** 📗 [02-05-cs4a.md](0-GUIDE/02-05-cs4a.md) 📗 [02-05a.md](0-GUIDE/02-05a.md)
  06. **Advanced Fine-tuning with HuggingFace** 📗 [02-06-cs4a.md](0-GUIDE/02-06-cs4a.md) 📗 [02-06a.md](0-GUIDE/02-06a.md)
  07. **LangChain – AI Application Development** 📗 [02-07-cs4a.md](0-GUIDE/02-07-cs4a.md) 📗 [02-07a.md](0-GUIDE/02-07a.md)
  08. **LangGraph (AI Agents)** 📗 [02-08-cs4a.md](0-GUIDE/02-08-cs4a.md) 📗 [02-08a.md](0-GUIDE/02-08a.md)
  09. **Semantic Kernel (AI Agents)** 📗 [02-09-cs4a.md](0-GUIDE/02-09-cs4a.md) 📗 [02-09a.md](0-GUIDE/02-09a.md)
  10. **Autogen (Advanced AI Agent Framework)** 📗 [02-10-cs4a.md](0-GUIDE/02-10-cs4a.md) 📗 [02-10a.md](0-GUIDE/02-10a.md)
  11. **AI Agent Development Workshop** 📗 [02-11-cs4a.md](0-GUIDE/02-11-cs4a.md) 📗 [02-11a.md](0-GUIDE/02-11a.md)
  12. **Summary and Future Directions** 📗 [02-12-cs4a.md](0-GUIDE/02-12-cs4a.md) 📗 [02-12a.md](0-GUIDE/02-12a.md)

- **3. AI Chatbot Development Course Guide** 📗 [03-ai-chatbots-en.md](0-GUIDE/03-ai-chatbots-en.md) <img src="https://flagcdn.com/w40/cz.png" width="15" alt="Czech Flag"> [03-ai-chatbots-cz.md](0-GUIDE/03-ai-chatbots-cz.md)
  01. **Introduction to AI Assistants and Creating Your First GPT Assistant** 📗 [03-01-cs4a.md](0-GUIDE/03-01-cs4a.md) 📗 [03-01a.md](0-GUIDE/03-01a.md), [03-01b.md](0-GUIDE/03-01b.md)
  02. **Capabilities and Limitations of GPT Assistants** 📗 [03-02-cs4a.md](0-GUIDE/03-02-cs4a.md) 📗 [03-02a.md](0-GUIDE/03-02a.md)
  03. **Vector Databases and Their Applications** 📗 [03-03-cs4a.md](0-GUIDE/03-03-cs4a.md) 📗 [03-03a.md](0-GUIDE/03-03a.md), [03-03b.md](0-GUIDE/03-03b.md)
  04. **Multi-Agent Orchestration with LangGraph** 📗 [03-04-cs4a.md](0-GUIDE/03-04-cs4a.md) 📗 [03-04a.md](0-GUIDE/03-04a.md)
  05. **Advanced API Integration for Dynamic Responses** 📗 [03-05-cs4a.md](0-GUIDE/03-05-cs4a.md) 📗 [03-05a.md](0-GUIDE/03-05a.md)
  06. **Monitoring and Performance Optimization** 📗 [03-06-cs4a.md](0-GUIDE/03-06-cs4a.md) 📗 [03-06a.md](0-GUIDE/03-06a.md)
  07. **Code Integration in GPT Assistant Responses** 📗 [03-07-cs4a.md](0-GUIDE/03-07-cs4a.md) 📗 [03-07a.md](0-GUIDE/03-07a.md)
  08. **Customer Assistant Design and Configuration** 📗 [03-08-cs4a.md](0-GUIDE/03-08-cs4a.md) 📗 [03-08a.md](0-GUIDE/03-08a.md)
  09. **Testing and Optimization of Customer Assistants** 📗 [03-09-cs4a.md](0-GUIDE/03-09-cs4a.md) 📗 [03-09a.md](0-GUIDE/03-09a.md)
  10. **Building Emotional Intelligence and Digital Twins** 📗 [03-10-cs4a.md](0-GUIDE/03-10-cs4a.md) 📗 [03-10a.md](0-GUIDE/03-10a.md)
  11. **Future Development Planning and Advanced Applications** 📗 [03-11-cs4a.md](0-GUIDE/03-11-cs4a.md) 📗 [03-11a.md](0-GUIDE/03-11a.md)

## 0. PROJECTS <small>(*Generated by [Claude Sonnet 4](https://www.anthropic.com/news/claude-4)*)</small>

- AI-LLM **MCP** *(Model Context Protocol)*
  
  - <small>Project selection by [Claude](https://claude.ai/) (01-01)</small>
    1. **Smart Code Assistant** (MCP, IDE integration, code analysis, Git repositories, documentation generation, TypeScript/Python)
    2. **Personal Knowledge Manager** (MCP, vector databases, RAG, document indexing, semantic search, note-taking, Obsidian/Notion integration)
    3. **Enterprise Data Analytics Dashboard** (MCP, SQL databases, business intelligence, data visualization, real-time reporting, PostgreSQL/MySQL)
    4. **Multi-Agent Customer Support System** (MCP, agent orchestration, ticket management, knowledge base integration, CRM systems, Zendesk/Salesforce)
    5. **Content Creation & SEO Optimizer** (MCP, web scraping, keyword analysis, content planning, social media APIs, WordPress/Shopify)
    6. **Financial Trading Assistant** (MCP, market data APIs, portfolio management, risk analysis, trading strategies, Bloomberg/Alpha Vantage)
    7. **Healthcare Documentation System** (MCP, medical records, FHIR standards, patient data analysis, compliance monitoring, HIPAA)
    8. **IoT Device Management Platform** (MCP, sensor networks, real-time monitoring, device control, MQTT, edge computing, AWS IoT)
    9. **Educational Learning Companion** (MCP, adaptive learning, progress tracking, curriculum management, LMS integration, personalized tutoring)
    10. **Automated DevOps Pipeline Manager** (MCP, CI/CD orchestration, infrastructure monitoring, deployment automation, Docker/Kubernetes, GitHub Actions)
  
  - <small>Project selection by [GPT](https://chatgpt.com/) (01-02)</small>
    1. **Legal Document Assistant**  
      *(MCP, LangChain, Pinecone, Retrieval-Augmented Generation, PDF Parsing, GPT-4o, Memory, Agents)*  
      → Upload legal documents (PDF), extract clauses, answer legal queries using structured context.
    2. **Enterprise Email Summarizer + Action Planner**  
      *(MCP, OpenAI GPT-4o, Context Windows, Task Memory, Prompt Engineering, LLM Agents)*  
      → Summarizes company emails, extracts action items, and plans tasks using MCP-based multi-turn context injection.
    3. **Medical Diagnosis and Treatment Advisor**  
      *(MCP, MedPrompt, LangGraph, OpenAI GPT-4o, Patient History Memory, Retrieval, JSON Templates)*  
      → Ingests structured patient data, retrieves similar cases, and generates diagnosis with justification.
    4. **Codebase Companion for Developers**  
      *(MCP, LangChain, GPT-4-turbo, VS Code Plugin, RAG, Embedding Index, Git Integration)*  
      → Interacts with local/global codebase using embeddings, tracks dev context with MCP layers for smarter Q&A.
    5. **Multi-Document Research Agent**  
      *(MCP, LangChain Agents, Multi-Modal RAG, Context Pruning, Pinecone, OpenAI Functions)*  
      → Allows uploading and querying across books, PDFs, articles, and websites with persistent context memory.
    6. **Customer Support Copilot (Multi-Turn Memory)**  
      *(MCP, LangGraph, GPT-4o, CRM Integration, Multi-Turn Chat Memory, Tools, Retrieval)*  
      → Handles long support sessions while maintaining context over past interactions using structured memory.
    7. **Financial Forecasting Chatbot**  
      *(MCP, Time-Series Data, LangChain, LLM + Tools, CSV Ingestion, Pandas Agent, GPT-4o)*  
      → Uses historical data, injects context using MCP for accurate financial forecasting and Q&A.
    8. **Multi-Agent Game Master (RPG)**  
      *(MCP, CrewAI, LangGraph, Character Memory, World Context, Function Calling, JSON Tools)*  
      → Game master for text-based RPGs using persistent character backstories and evolving world state.
    9. **Scientific Paper Translator & Explainer**  
      *(MCP, Arxiv API, LangChain, Translator Tool, Citation Retriever, Summary Memory)*  
      → Parses scientific papers and translates/explains them using retrieved context, citations, and summary layers.
    10. **Interactive Education Tutor (All Subjects)**  
      *(MCP, LangChain, Long-Term Memory, Concept Tracking, OpenAI GPT-4o, Gradio UI)*  
      → Personalized tutor using MCP to track concepts learned, context windows per subject, and goal tracking.
  
  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (01-03)</small>
    1. **MCP-Powered Dynamic Chatbot Framework**  
      (*MCP, Conversational AI, Context-Aware Dialogs, RAG, Fine-Tuning, LangChain*)
    2. **Automated Legal Document Analyzer with MCP**  
      (*MCP, NLP, Contract Review, Semantic Search, Zero-Shot Classification, Haystack*)
    3. **Personalized Learning Assistant Using MCP**  
      (*MCP, Adaptive Education, Knowledge Graphs, GPT-4 Integration, Feedback Loops*)
    4. **MCP-Based Multi-Agent Simulation Environment**  
      (*MCP, Autonomous Agents, Game Theory, Emergent Behavior, LLM Orchestration*)
    5. **Enterprise Knowledge Management with MCP**  
      (*MCP, Vector Databases, Document Embeddings, Retrieval-Augmented Generation, LlamaIndex*)
    6. **AI-Powered Research Paper Summarization (MCP-Driven)**  
      (*MCP, Academic NLP, Abstractive Summarization, Citation Graphs, BERTopic*)
    7. **Context-Aware Code Generation & Debugging Tool**  
      (*MCP, GitHub Copilot Alternative, AST Parsing, Error Correction, MCP-Guided Prompts*)
    8. **MCP-Enhanced Multilingual Translation System**  
      (*MCP, Low-Resource Languages, Context Retention, LLM Alignment, OPUS Models*)
    9. **Dynamic Storytelling & Interactive Fiction Platform**  
      (*MCP, Narrative Generation, User-Driven Plotlines, MCP Memory Layers, Stable Diffusion*)
    10. **MCP-Optimized Fraud Detection for Financial Transactions**  
        (*MCP, Anomaly Detection, Graph Neural Networks, Explainable AI, Synthetic Data*)

- AI-LLM **Multi-Agent Systems**

  - <small>Project selection by [Claude](https://claude.ai/) (02-01)</small>
    1. **Autonomous Trading Floor Simulation** (Financial Markets, Risk Management, Portfolio Optimization, Real-time Decision Making, Market Analysis Agents)
    2. **Smart City Traffic Management System** (IoT Integration, Traffic Flow Optimization, Emergency Response Coordination, Predictive Analytics, Urban Planning Agents)
    3. **Collaborative Research Assistant Network** (Knowledge Discovery, Literature Review, Hypothesis Generation, Peer Review, Academic Research Coordination)
    4. **Distributed Customer Service Ecosystem** (Natural Language Processing, Sentiment Analysis, Ticket Routing, Escalation Management, Multi-channel Support)
    5. **Supply Chain Optimization Platform** (Logistics Coordination, Inventory Management, Demand Forecasting, Supplier Negotiation, Risk Assessment Agents)
    6. **Personalized Learning Management System** (Adaptive Learning, Content Curation, Student Assessment, Progress Tracking, Educational Resource Coordination)
    7. **Healthcare Diagnosis and Treatment Planning** (Medical Knowledge Integration, Symptom Analysis, Treatment Recommendation, Drug Interaction Checking, Clinical Decision Support)
    8. **Real Estate Market Analysis and Recommendation Engine** (Property Valuation, Market Trend Analysis, Investment Advisory, Neighborhood Assessment, Client Matching)
    9. **Cybersecurity Threat Detection and Response Network** (Anomaly Detection, Threat Intelligence, Incident Response, Vulnerability Assessment, Security Policy Enforcement)
    10. **Content Creation and Marketing Automation Hub** (Content Strategy, SEO Optimization, Social Media Management, Campaign Performance Analysis, Brand Voice Consistency)

  - <small>Project selection by [GPT](https://chatgpt.com/) (02-02)</small>
    1. **Autonomous Research Team Assistant**  
      *(RAG, LangChain Agents, Pinecone, OpenAI GPT-4o, Task Decomposition, Memory, Tool Use, Web Search)*  
      → Multi-agent system that autonomously performs academic research, dividing tasks among agents (e.g., summarizer, fact-checker, citation retriever).
    2. **AI Legal Document Analyzer**  
      *(LLM Agents, Document QA, Retrieval, Legal NLP, Memory Graphs, Semantic Search, Claude Sonnet, LangGraph)*  
      → Team of legal-AI agents parsing legal documents to extract clauses, detect risks, and recommend actions.
    3. **Financial Market Analysis and Decision Support**  
      *(Multi-Agent Collaboration, Real-Time Data, Forecasting, News Scraping, Trading APIs, Agent Communication, OpenAI + Anthropic)*  
      → Agents specialize in trends, risk analysis, news sentiment, and portfolio balancing for human financial advisors.
    4. **AI Game Master for Text-Based RPGs**  
      *(LLM Agents, Role-Based Reasoning, Memory, Game State Engine, Prompt Engineering, React Framework, Text-to-Action)*  
      → Game master and multiple character agents co-create an evolving narrative in an interactive RPG.
    5. **Customer Support AI Team for E-Commerce**  
      *(LangChain, Vectorstore Retrieval, Agent Hand-off, LLM Tool Use, E-Commerce API, Multi-LLM Setup)*  
      → Agents for order tracking, returns, product info, and escalation working in coordination to resolve customer queries.
    6. **Scientific Paper Reviewer Committee**  
      *(Autonomous Agents, Role-Playing Reviewers, LLM+Tool Use, Citation Verification, Markdown Reports, Critique Chains)*  
      → Reviewer agents simulate peer review with feedback, scoring, and revision suggestions for academic submissions.
    7. **Smart City Infrastructure Management**  
      *(Sensor Data Simulation, AI Agents, Planning, Multi-Agent Communication, UMAP/GraphQL, Decision Making)*  
      → Agents oversee traffic, energy, and safety systems—collaborating to optimize city-wide resources and alerts.
    8. **AI Debate Team**  
      *(Multi-Agent Prompt Chaining, Role-Specific LLMs, Memory, Persuasion Logic, GPT-4o, Claude, Open Debate APIs)*  
      → Pro and con agents prepare and argue on a topic in structured debate format with live audience Q&A moderation.
    9. **AI Software Engineering Team**  
      *(LLM Agents for Code Writing, Code Review, Debugging, Test Generation, GitHub API, LangGraph, Code Interpreter)*  
      → Agents emulate developer roles (coder, reviewer, tester) working on a collaborative software project.
    10. **Multi-Agent Travel Planner**  
      *(Agents for Flight, Hotel, Activity, Budget Planning, External API Calls, Natural Language Queries, User Preference Memory)*  
      → A team of AI agents that negotiate and compile optimal travel plans based on user goals and constraints.

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (02-03)</small>
    1. **Autonomous Research Team**  
      (*LLM agents, collaborative research, semantic search, knowledge synthesis, Python, LangChain*)  
      A multi-agent system where AI researchers autonomously gather, analyze, and summarize academic papers.
    2. **Smart Contract Auditor Squad**  
      (*Blockchain, Solidity, LLM agents, vulnerability detection, collaborative reporting*)  
      AI agents simulate a security team auditing smart contracts, identifying bugs and suggesting fixes.
    3. **AI-Powered Debate Platform**  
      (*Multi-agent debate, argument synthesis, truth discovery, reinforcement learning*)  
      LLM agents with opposing viewpoints debate complex topics to uncover balanced conclusions.
    4. **Dynamic Customer Support Swarm**  
      (*Multi-agent helpdesk, intent routing, live collaboration, CRM integration*)  
      AI agents work together to handle customer queries, escalating and sharing context as needed.
    5. **Generative Game Design Collective**  
      (*Procedural content generation, LLM agents, Unity/Unreal, collaborative creativity*)  
      Autonomous AI designers, writers, and testers collaborate to create game narratives and mechanics.
    6. **AI Venture Capital Simulator**  
      (*Market analysis, startup evaluation, multi-agent decision-making, financial modeling*)  
      A simulated VC firm where AI agents scout, evaluate, and negotiate startup investments.
    7. **Epidemic Response Coordinator**  
      (*Public health, data analysis, policy simulation, multi-agent cooperation*)  
      AI agents model disease spread, evaluate interventions, and optimize resource allocation.
    8. **Legal Case Strategizer**  
      (*Legal AI, precedent analysis, argument generation, multi-perspective evaluation*)  
      A virtual law firm where AI agents develop case strategies by analyzing past rulings.
    9. **Personalized Education Collective**  
      (*Adaptive learning, LLM tutors, multi-agent curriculum design, knowledge assessment*)  
      AI tutors with different specialties collaborate to create personalized learning paths.
    10. **AI Film Production Studio**  
        (*Multi-modal agents, script generation, storyboarding, virtual production*)  
        Autonomous AI teams handle screenwriting, casting, and editing for generated short films.

- AI-LLM **RAG** *(Retrieval-Augmented Generation)*

  - <small>Project selection by [Claude](https://claude.ai/) (03-01)</small>
    1. **Intelligent Document Assistant**
    (RAG, Document Processing, PDF Parsing, Vector Embeddings, Semantic Search, OpenAI GPT, Langchain, FAISS)
    2. **Enterprise Knowledge Base Chatbot**
    (RAG, Internal Documentation, Employee Support, Confluence Integration, Azure OpenAI, Pinecone, Slack Bot API)
    3. **Legal Research and Case Analysis Platform**
    (RAG, Legal Documents, Case Law, Regulatory Compliance, Elasticsearch, Claude API, Citation Extraction, NER)
    4. **Medical Literature Review System**
    (RAG, PubMed Integration, Clinical Papers, Drug Information, BioBERT, Chroma DB, Medical NLP, HIPAA Compliance)
    5. **Code Documentation and Bug Resolution Assistant**
    (RAG, GitHub Integration, Stack Overflow, Technical Documentation, Code Embeddings, Weaviate, GitHub Copilot API)
    6. **Financial Research and Investment Analysis Tool**
    (RAG, SEC Filings, Financial Reports, Market Data, Time-series Analysis, Qdrant, Bloomberg API, Risk Assessment)
    7. **E-learning Content Recommendation Engine**
    (RAG, Course Materials, Educational Videos, Learning Paths, Content Similarity, Milvus, YouTube API, Adaptive Learning)
    8. **Customer Support Knowledge Assistant**
    (RAG, FAQ Database, Ticket History, Product Manuals, Sentiment Analysis, Redis Search, Zendesk Integration, Multi-language)
    9. **Scientific Research Paper Discovery Platform**
    (RAG, ArXiv Papers, Research Abstracts, Citation Networks, SciBERT, Neo4j, Graph RAG, Collaborative Filtering)
    10. **Real Estate Property Intelligence System**
    (RAG, Property Listings, Market Reports, Neighborhood Data, Geospatial Search, PostGIS, Zillow API, Location Embeddings)

  - <small>Project selection by [GPT](https://chatgpt.com/) (03-02)</small>
    1. **Legal Document Assistant**  
      *(RAG, LangChain, OpenAI GPT-4o, Pinecone, PDF Parsing, Prompt Engineering, Semantic Search)*  
      → AI assistant for answering legal questions from a repository of contracts and legal documents.
    2. **Medical Research Summarizer**  
      *(RAG, LlamaIndex, PubMed API, HuggingFace Transformers, BioBERT, Vector Databases, Summarization)*  
      → Summarizes and answers questions using biomedical research papers and studies.
    3. **Enterprise Knowledge Chatbot**  
      *(RAG, LangChain, Azure OpenAI, SharePoint, FAISS, Authentication, Multi-user Chat)*  
      → Internal chatbot that helps employees access company documents and SOPs securely.
    4. **Academic Tutor for Students**  
      *(RAG, Claude Haiku, PDF + DOCX ingestion, Milvus, Streamlit, Flashcards, Quiz Generator)*  
      → Personalized study assistant that extracts content from textbooks and generates quizzes.
    5. **Codebase Q&A Bot**  
      *(RAG, OpenAI GPT-4o, GitHub API, Docstring Parsing, Embeddings, VSCode Extension)*  
      → AI bot that answers technical questions from your codebase and documentation files.
    6. **Multilingual Travel Assistant**  
      *(RAG, GPT-4 Turbo, Translation APIs, Pinecone, CSV/JSON data, Location-based RAG)*  
      → Answers tourist queries using local attraction guides and multilingual content.
    7. **Historical Archive Explorer**  
      *(RAG, Llama 3, OCR, Newspapers Archive, Vector Search, Timeline Generation)*  
      → Allows users to explore and query historical texts, scanned papers, and rare books.
    8. **Customer Support Ticket Analyzer**  
      *(RAG, LangChain, ElasticSearch, Email Parsing, OpenAI Functions, Retrieval Filters)*  
      → AI that assists support teams by answering and classifying tickets using historical data.
    9. **Scientific Paper Explorer for Engineers**  
      *(RAG, LlamaIndex, ArXiv API, Embedding Comparison, Visual Search, Streamlit UI)*  
      → Engineers can query technical papers and extract specific methods or evaluations.
    10. **Legal & Regulatory Compliance Checker**  
      *(RAG, Claude Sonnet, Regulatory PDFs, Pinecone Hybrid Search, LangChain Agents)*  
      → Helps companies verify if internal policies comply with external laws and standards.

  - <small>Project selection by [DeepSeek](https://chat.deepseek.com/) (03-03)</small>