import streamlit as st
from autogen import (
    SwarmAgent,
    SwarmResult,
    initiate_swarm_chat,
    AFTER_WORK,
    UPDATE_SYSTEM_MESSAGE,
)
from agent_utils import update_system_message_func  # import helper function
import json

# Initialize session state
if "output" not in st.session_state:
    # add "FIRST" and "LAST" to session state
    st.session_state.output = {
        # "story": "", "gameplay": "", "visuals": "", "tech": "", 
        "FIRST": "", "story": "", "gameplay": "", "visuals": "", "tech": "", "LAST": "",
        }

# Sidebar for API key input
st.sidebar.title("API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Add guidance in sidebar
st.sidebar.success(
    """
✨ **Getting Started**

Please provide inputs and features for your dream game! Consider:
- The overall vibe and setting
- Core gameplay elements
- Target audience and platforms
- Visual style preferences
- Technical requirements

The AI agents will collaborate to develop a comprehensive game concept based on your specifications.
"""
)

# Main app UI
st.title("🎮 AI Game Design Agent Team")

# Add agent information below title
st.info(
    """
**Meet Your AI Game Design Team:**

🎭 **Story Agent** - Crafts compelling narratives and rich worlds

🎮 **Gameplay Agent** - Creates engaging mechanics and systems

🎨 **Visuals Agent** - Shapes the artistic vision and style

⚙️ **Tech Agent** - Provides technical direction and solutions

These agents collaborate to create a comprehensive game concept based on your inputs.
"""
)

# User inputs
st.subheader("Game Details")
col1, col2 = st.columns(2)

with col1:
    background_vibe = st.text_input("Background Vibe", "Epic fantasy with dragons")
    game_type = st.selectbox(
        "Game Type",
        [
            "RPG",
            "Action",
            "Adventure",
            "Puzzle",
            "Strategy",
            "Simulation",
            "Platform",
            "Horror",
        ],
    )
    target_audience = st.selectbox(
        "Target Audience",
        [
            "Kids (7-12)",
            "Teens (13-17)",
            "Young Adults (18-25)",
            "Adults (26+)",
            "All Ages",
        ],
    )
    player_perspective = st.selectbox(
        "Player Perspective",
        ["First Person", "Third Person", "Top Down", "Side View", "Isometric"],
    )
    multiplayer = st.selectbox(
        "Multiplayer Support",
        [
            "Single Player Only",
            "Local Co-op",
            "Online Multiplayer",
            "Both Local and Online",
        ],
    )

with col2:
    game_goal = st.text_input("Game Goal", "Save the kingdom from eternal winter")
    art_style = st.selectbox(
        "Art Style",
        [
            "Realistic",
            "Cartoon",
            "Pixel Art",
            "Stylized",
            "Low Poly",
            "Anime",
            "Hand-drawn",
        ],
    )
    platform = st.multiselect(
        "Target Platforms",
        ["PC", "Mobile", "PlayStation", "Xbox", "Nintendo Switch", "Web Browser"],
    )
    development_time = st.slider("Development Time (months)", 1, 36, 12)
    cost = st.number_input("Budget (USD)", min_value=0, value=10000, step=5000)

# Additional details
st.subheader("Detailed Preferences")
col3, col4 = st.columns(2)

with col3:
    "col3"
    core_mechanics = st.multiselect(
        "Core Gameplay Mechanics",
        [
            "Combat",
            "Exploration",
            "Puzzle Solving",
            "Resource Management",
            "Base Building",
            "Stealth",
            "Racing",
            "Crafting",
        ],
    )
    mood = st.multiselect(
        "Game Mood/Atmosphere",
        [
            "Epic",
            "Mysterious",
            "Peaceful",
            "Tense",
            "Humorous",
            "Dark",
            "Whimsical",
            "Scary",
        ],
    )

with col4:
    "col4"
    inspiration = st.text_area("Games for Inspiration (comma-separated)", "")
    unique_features = st.text_area("Unique Features or Requirements", "")

depth = st.selectbox("Level of Detail in Response", ["Low", "Medium", "High"])

# Button to start the agent collaboration
if st.button("Generate Game Concept"):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        with st.spinner("🤖 AI Agents are collaborating on your game concept..."):
            # Prepare the task based on user inputs
            task = f"""
            Create a game concept with the following details:
            - Background Vibe: {background_vibe}
            - Game Goal: {game_goal}
            - Final answer in English, Spanish and Czech language (return markdown list en, es, vn, cz).
            """

            llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": api_key}]}

            # initialize context variables
            context_variables = {
                "story": None,
                "gameplay": None,
                "visuals": None,
                "tech": None,
            }

            # define functions to be called by the agents
            def update_story_overview(
                story_summary: str, context_variables: dict
            ) -> SwarmResult:
                """Keep the summary as short as possible."""
                context_variables["story"] = story_summary
                st.sidebar.success("Story overview: " + story_summary)
                return SwarmResult(
                    agent="gameplay_agent", context_variables=context_variables
                )

            def update_gameplay_overview(
                gameplay_summary: str, context_variables: dict
            ) -> SwarmResult:
                """Keep the summary as short as possible."""
                context_variables["gameplay"] = gameplay_summary
                st.sidebar.success("Gameplay overview: " + gameplay_summary)
                return SwarmResult(
                    agent="visuals_agent", context_variables=context_variables
                )

            def update_visuals_overview(
                visuals_summary: str, context_variables: dict
            ) -> SwarmResult:
                """Keep the summary as short as possible."""
                context_variables["visuals"] = visuals_summary
                st.sidebar.success("Visuals overview: " + visuals_summary)
                return SwarmResult(
                    agent="tech_agent", context_variables=context_variables
                )

            def update_tech_overview(
                tech_summary: str, context_variables: dict
            ) -> SwarmResult:
                """Keep the summary as short as possible."""
                context_variables["tech"] = tech_summary
                st.sidebar.success("Tech overview: " + tech_summary)
                return SwarmResult(
                    agent="story_agent", context_variables=context_variables
                )

            state_update = UPDATE_SYSTEM_MESSAGE(update_system_message_func)

            # Define agents
            story_agent = SwarmAgent(
                "story_agent",
                llm_config=llm_config,
                functions=update_story_overview,
                update_agent_state_before_reply=[state_update],
            )

            gameplay_agent = SwarmAgent(
                "gameplay_agent",
                llm_config=llm_config,
                functions=update_gameplay_overview,
                update_agent_state_before_reply=[state_update],
            )

            visuals_agent = SwarmAgent(
                "visuals_agent",
                llm_config=llm_config,
                functions=update_visuals_overview,
                update_agent_state_before_reply=[state_update],
            )

            tech_agent = SwarmAgent(
                name="tech_agent",
                llm_config=llm_config,
                functions=update_tech_overview,
                update_agent_state_before_reply=[state_update],
            )

            result, _, _ = initiate_swarm_chat(
                initial_agent=story_agent,
                agents=[story_agent, gameplay_agent, visuals_agent, tech_agent],
                user_agent=None,
                messages=task,
                max_rounds=13,
            )

            print("len(result)", len(result.chat_history))
            # print("result", result.chat_history[0]["content"])
            # print("result", result.chat_history[1]["content"])
            # print("result", result.chat_history[2]["content"])
            # print("result", result.chat_history[3]["content"])
            # print("result", result.chat_history[4]["content"])
            # refactore to use for loop
            # for i in range(len(result.chat_history)):
            #     # print("result", result.chat_history[i]["content"])
            #     print("result", result.chat_history[i].keys())
            # for i in range(len(result.chat_history)):
            #     role = result.chat_history[i]["role"]
                # name = result.chat_history[i]["name"]
                # content = result.chat_history[i]["content"]
                # print("result", result.chat_history[i]["content"])

            # print("result", result.chat_history)

#             chat_history = [  # Example data extracted from the file
#     {'content': '...', 'role': 'assistant', 'name': 'story_agent'},
#     {'content': 'None', 'tool_calls': [{'id': 'call_...', 'function': {'arguments': '{"story_summary":"..."}}'}], 'name': 'story_agent', 'role': 'assistant'},
#     {'content': '', 'tool_responses': [{'tool_call_id': 'call_...', 'role': 'tool', 'content': ''}], 'name': '_Swarm_Tool_Executor', 'role': 'tool'},
#     {'content': '...', 'role': 'user', 'name': 'story_agent'}
# ]
            # chat_history = result.chat_history
            # # Extract messages from assistant agents
            # agent_messages = [entry['content'] for entry in chat_history if entry['role'] == 'assistant' and entry['content']]

            # for msg in agent_messages:
            #     print(msg)
            # Extract first and last content
            chat_history = result.chat_history
            first_content = next((msg['content'] for msg in chat_history if msg['content']), None)
            last_content = next((msg['content'] for msg in reversed(chat_history) if msg['content']), None)

            # Extract summaries
            summaries = {}
            for msg in chat_history:
                if 'tool_calls' in msg:
                    for tool_call in msg['tool_calls']:
                        arguments = json.loads(tool_call['function']['arguments'])
                        for key in ["story_summary", "gameplay_summary", "visuals_summary", "tech_summary"]:
                            if key in arguments:
                                summaries[key] = arguments[key]

            # Print results
            print(f"First Content: {first_content}")
            print(f"Last Content: {last_content}")
            print("Summaries:")
            for key, value in summaries.items():
                print(f"- {key}: {value}")


# [RESULT] result.chat_history[i].keys()
# result dict_keys(['content', 'role'])
# result dict_keys(['content', 'tool_calls', 'name', 'role'])
# result dict_keys(['content', 'tool_responses', 'name', 'role'])
# result dict_keys(['content', 'tool_calls', 'name', 'role'])
# result dict_keys(['content', 'tool_responses', 'name', 'role'])
# result dict_keys(['content', 'tool_calls', 'name', 'role'])
# result dict_keys(['content', 'tool_responses', 'name', 'role'])
# result dict_keys(['content', 'tool_calls', 'name', 'role'])
# result dict_keys(['content', 'tool_responses', 'name', 'role'])
# result dict_keys(['content', 'name', 'role'])


            # Update session state with the individual responses
            st.session_state.output = {
                "story": result.chat_history[-4]["content"],
                "gameplay": result.chat_history[-3]["content"],
                "visuals": result.chat_history[-2]["content"],
                "tech": result.chat_history[-1]["content"],
                "FIRST": first_content,
                "story": summaries["story_summary"],
                "gameplay": summaries["gameplay_summary"],
                "visuals": summaries["visuals_summary"],
                "tech": summaries["tech_summary"],
                "LAST": last_content,
                  
            }

        # Display success message after completion
        st.success("✨ Game concept generated successfully!")

        # Display the individual outputs in expanders
        with st.expander("First Content"):
            st.markdown(st.session_state.output["FIRST"])

        with st.expander("Story Design"):
            st.markdown(st.session_state.output["story"])

        with st.expander("Gameplay Mechanics"):
            st.markdown(st.session_state.output["gameplay"])

        with st.expander("Visual and Audio Design"):
            st.markdown(st.session_state.output["visuals"])

        with st.expander("Technical Recommendations"):
            st.markdown(st.session_state.output["tech"])

        with st.expander("Last Content"):
            st.markdown(st.session_state.output["LAST"])

# import streamlit as st
# from autogen import (
#     SwarmAgent,
#     SwarmResult,
#     initiate_swarm_chat,
#     AFTER_WORK,
#     UPDATE_SYSTEM_MESSAGE,
# )
# from agent_utils import update_system_message_func  # import helper function

# # Initialize session state
# if "output" not in st.session_state:
#     st.session_state.output = {"story": "", "gameplay": "", "visuals": "", "tech": ""}

# # Sidebar for API key input
# st.sidebar.title("API Key")
# api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# # Add guidance in sidebar
# st.sidebar.success(
#     """
# ✨ **Getting Started**

# Please provide inputs and features for your dream game! Consider:
# - The overall vibe and setting
# - Core gameplay elements
# - Target audience and platforms
# - Visual style preferences
# - Technical requirements

# The AI agents will collaborate to develop a comprehensive game concept based on your specifications.
# """
# )

# # Main app UI
# st.title("🎮 AI Game Design Agent Team")

# # Add agent information below title
# st.info(
#     """
# **Meet Your AI Game Design Team:**

# 🎭 **Story Agent** - Crafts compelling narratives and rich worlds

# 🎮 **Gameplay Agent** - Creates engaging mechanics and systems

# 🎨 **Visuals Agent** - Shapes the artistic vision and style

# ⚙️ **Tech Agent** - Provides technical direction and solutions

# These agents collaborate to create a comprehensive game concept based on your inputs.
# """
# )

# # User inputs
# st.subheader("Game Details")
# col1, col2 = st.columns(2)

# with col1:
#     background_vibe = st.text_input("Background Vibe", "Epic fantasy with dragons")
#     game_type = st.selectbox(
#         "Game Type",
#         [
#             "RPG",
#             "Action",
#             "Adventure",
#             "Puzzle",
#             "Strategy",
#             "Simulation",
#             "Platform",
#             "Horror",
#         ],
#     )
#     target_audience = st.selectbox(
#         "Target Audience",
#         [
#             "Kids (7-12)",
#             "Teens (13-17)",
#             "Young Adults (18-25)",
#             "Adults (26+)",
#             "All Ages",
#         ],
#     )
#     player_perspective = st.selectbox(
#         "Player Perspective",
#         ["First Person", "Third Person", "Top Down", "Side View", "Isometric"],
#     )
#     multiplayer = st.selectbox(
#         "Multiplayer Support",
#         [
#             "Single Player Only",
#             "Local Co-op",
#             "Online Multiplayer",
#             "Both Local and Online",
#         ],
#     )

# with col2:
#     game_goal = st.text_input("Game Goal", "Save the kingdom from eternal winter")
#     art_style = st.selectbox(
#         "Art Style",
#         [
#             "Realistic",
#             "Cartoon",
#             "Pixel Art",
#             "Stylized",
#             "Low Poly",
#             "Anime",
#             "Hand-drawn",
#         ],
#     )
#     platform = st.multiselect(
#         "Target Platforms",
#         ["PC", "Mobile", "PlayStation", "Xbox", "Nintendo Switch", "Web Browser"],
#     )
#     development_time = st.slider("Development Time (months)", 1, 36, 12)
#     cost = st.number_input("Budget (USD)", min_value=0, value=10000, step=5000)

# # Additional details
# st.subheader("Detailed Preferences")
# col3, col4 = st.columns(2)

# with col3:
#     core_mechanics = st.multiselect(
#         "Core Gameplay Mechanics",
#         [
#             "Combat",
#             "Exploration",
#             "Puzzle Solving",
#             "Resource Management",
#             "Base Building",
#             "Stealth",
#             "Racing",
#             "Crafting",
#         ],
#     )
#     mood = st.multiselect(
#         "Game Mood/Atmosphere",
#         [
#             "Epic",
#             "Mysterious",
#             "Peaceful",
#             "Tense",
#             "Humorous",
#             "Dark",
#             "Whimsical",
#             "Scary",
#         ],
#     )

# with col4:
#     inspiration = st.text_area("Games for Inspiration (comma-separated)", "")
#     unique_features = st.text_area("Unique Features or Requirements", "")

# depth = st.selectbox("Level of Detail in Response", ["Low", "Medium", "High"])

# # Button to start the agent collaboration
# if st.button("Generate Game Concept"):
#     # Check if API key is provided
#     if not api_key:
#         st.error("Please enter your OpenAI API key.")
#     else:
#         with st.spinner("🤖 AI Agents are collaborating on your game concept..."):
#             # Prepare the task based on user inputs
#             task = f"""
#             Create a game concept with the following details:
#             - Background Vibe: {background_vibe}
#             - Game Type: {game_type}
#             - Game Goal: {game_goal}
#             - Target Audience: {target_audience}
#             - Player Perspective: {player_perspective}
#             - Multiplayer Support: {multiplayer}
#             - Art Style: {art_style}
#             - Target Platforms: {', '.join(platform)}
#             - Development Time: {development_time} months
#             - Budget: ${cost:,}
#             - Core Mechanics: {', '.join(core_mechanics)}
#             - Mood/Atmosphere: {', '.join(mood)}
#             - Inspiration: {inspiration}
#             - Unique Features: {unique_features}
#             - Detail Level: {depth}
#             """

#             llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": api_key}]}

#             # initialize context variables
#             context_variables = {
#                 "story": None,
#                 "gameplay": None,
#                 "visuals": None,
#                 "tech": None,
#             }

#             # define functions to be called by the agents
#             def update_story_overview(
#                 story_summary: str, context_variables: dict
#             ) -> SwarmResult:
#                 """Keep the summary as short as possible."""
#                 context_variables["story"] = story_summary
#                 st.sidebar.success("Story overview: " + story_summary)
#                 return SwarmResult(
#                     agent="gameplay_agent", context_variables=context_variables
#                 )

#             def update_gameplay_overview(
#                 gameplay_summary: str, context_variables: dict
#             ) -> SwarmResult:
#                 """Keep the summary as short as possible."""
#                 context_variables["gameplay"] = gameplay_summary
#                 st.sidebar.success("Gameplay overview: " + gameplay_summary)
#                 return SwarmResult(
#                     agent="visuals_agent", context_variables=context_variables
#                 )

#             def update_visuals_overview(
#                 visuals_summary: str, context_variables: dict
#             ) -> SwarmResult:
#                 """Keep the summary as short as possible."""
#                 context_variables["visuals"] = visuals_summary
#                 st.sidebar.success("Visuals overview: " + visuals_summary)
#                 return SwarmResult(
#                     agent="tech_agent", context_variables=context_variables
#                 )

#             def update_tech_overview(
#                 tech_summary: str, context_variables: dict
#             ) -> SwarmResult:
#                 """Keep the summary as short as possible."""
#                 context_variables["tech"] = tech_summary
#                 st.sidebar.success("Tech overview: " + tech_summary)
#                 return SwarmResult(
#                     agent="story_agent", context_variables=context_variables
#                 )

#             state_update = UPDATE_SYSTEM_MESSAGE(update_system_message_func)

#             # Define agents
#             story_agent = SwarmAgent(
#                 "story_agent",
#                 llm_config=llm_config,
#                 functions=update_story_overview,
#                 update_agent_state_before_reply=[state_update],
#             )

#             gameplay_agent = SwarmAgent(
#                 "gameplay_agent",
#                 llm_config=llm_config,
#                 functions=update_gameplay_overview,
#                 update_agent_state_before_reply=[state_update],
#             )

#             visuals_agent = SwarmAgent(
#                 "visuals_agent",
#                 llm_config=llm_config,
#                 functions=update_visuals_overview,
#                 update_agent_state_before_reply=[state_update],
#             )

#             tech_agent = SwarmAgent(
#                 name="tech_agent",
#                 llm_config=llm_config,
#                 functions=update_tech_overview,
#                 update_agent_state_before_reply=[state_update],
#             )

#             # story_agent.register_hand_off(AFTER_WORK(gameplay_agent))
#             # gameplay_agent.register_hand_off(AFTER_WORK(visuals_agent))
#             # visuals_agent.register_hand_off(AFTER_WORK(tech_agent))
#             # tech_agent.register_hand_off(AFTER_WORK(story_agent))

#             # story_agent.register_hook(AFTER_WORK(gameplay_agent))
#             # gameplay_agent.register_hook(AFTER_WORK(visuals_agent))
#             # visuals_agent.register_hook(AFTER_WORK(tech_agent))
#             # tech_agent.register_hook(AFTER_WORK(story_agent))

#             # story_agent.register_hook(trigger=AFTER_WORK, hook=lambda: gameplay_agent)
#             # gameplay_agent.register_hook(trigger=AFTER_WORK, hook=lambda: visuals_agent)
#             # visuals_agent.register_hook(trigger=AFTER_WORK, hook=lambda: tech_agent)
#             # tech_agent.register_hook(trigger=AFTER_WORK, hook=lambda: story_agent)

#             # story_agent.register_hook(AFTER_WORK, lambda sender, recipient, message, context: gameplay_agent)
#             # gameplay_agent.register_hook(AFTER_WORK, lambda sender, recipient, message, context: visuals_agent)
#             # visuals_agent.register_hook(AFTER_WORK, lambda sender, recipient, message, context: tech_agent)
#             # tech_agent.register_hook(AFTER_WORK, lambda sender, recipient, message, context: story_agent)


#             # result, _, _ = initiate_swarm_chat(
#             #     initial_agent=story_agent,
#             #     agents=[story_agent, gameplay_agent, visuals_agent, tech_agent],
#             #     user_agent=None,
#             #     messages=task,
#             #     max_rounds=13,
#             # )
#             result, _, _ = initiate_swarm_chat(
#                 initial_agent=story_agent,
#                 agents=[story_agent, gameplay_agent, visuals_agent, tech_agent],
#                 user_agent=None,
#                 messages=task,
#                 max_rounds=13,
#             )

#             # Update session state with the individual responses
#             st.session_state.output = {
#                 "story": result.chat_history[-4]["content"],
#                 "gameplay": result.chat_history[-3]["content"],
#                 "visuals": result.chat_history[-2]["content"],
#                 "tech": result.chat_history[-1]["content"],
#             }

#         # Display success message after completion
#         st.success("✨ Game concept generated successfully!")

#         # Display the individual outputs in expanders
#         with st.expander("Story Design"):
#             st.markdown(st.session_state.output["story"])

#         with st.expander("Gameplay Mechanics"):
#             st.markdown(st.session_state.output["gameplay"])

#         with st.expander("Visual and Audio Design"):
#             st.markdown(st.session_state.output["visuals"])

#         with st.expander("Technical Recommendations"):
#             st.markdown(st.session_state.output["tech"])
