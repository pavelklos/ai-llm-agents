<hr>

`ag2-groupchat.ipynb`
`ag2-swarm.ipynb`

**AG2** offers two main approaches for multi-agent systems:
- **GroupChat**
- **Swarm**

Key differences:
- GroupChat uses a manager to select the next speaker
- Swarm uses explicit handoffs between agents
- Swarm maintains shared context variables across agents

Choose:
- GroupChat for collaborative discussions
- Swarm for more structured workflows with clear handoff points

<hr>

`ag2-swarm-streamlit.py`

This Streamlit app creates a recipe generation system using AG2 swarm agents.<br>
Here's what it includes:
1. Three specialized agents:
   - Recipe Planner: Plans ingredients and structure
   - Chef: Creates detailed cooking instructions
   - Nutritionist: Adds health information and tips
2. UI Features:
   - API key input (from .env or manual entry)
   - Model selection and temperature control
   - Recipe customization (cuisine, dietary restrictions, cooking time)
   - Progress tracking during generation
   - Tabbed interface to view each agent's contribution
   - Download button for the final recipe
   - Rating system for feedback
   - Expandable sections for additional information
3. Swarm Implementation:
   - Structured workflow with explicit handoffs between agents
   - Context variables to share information between agents
   - Custom functions to record each agent's contribution
   - Final recipe combining all agents' work
4. Streamlit Components:
   - Tabs, expanders, and columns for layout
   - Progress bar and status updates
   - Sliders, selectors, and text inputs
   - Custom CSS for better styling
   - Download button for saving results
   - Interactive rating system

To run this app:<br>
`streamlit run ag2-swarm-streamlit.py`<br>
Make sure you have the required packages installed:<br>
`streamlit, autogen, python-dotenv`