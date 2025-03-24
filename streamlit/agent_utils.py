from autogen import SwarmAgent, OpenAIWrapper

# system_messages = {
#     "story_agent": """
# Design the game's narrative, characters, world-building, and story progression to create an engaging and immersive experience.
# Answer in 50 words or less.
#     """,
#     "gameplay_agent": """
# Design the core mechanics, progression systems, player interactions, and game modes to create engaging and balanced.
# Answer in 50 words or less.
#     """,
#     "visuals_agent": """
# Design the visual style, characters, environments, animations, and audio elements to create an immersive and visually appealing.
# Answer in 50 words or less.
#     """,
#     "tech_agent": """
# Recommend the game engine, technical requirements, development pipeline, asset workflow, and scalability to ensure a smooth development process.
# Answer in 50 words or less.
#     """,
# }
system_messages = {
    "story_agent": """
You are an experienced game story designer specializing in narrative design and world-building. Your task is to:
1. Create a compelling narrative that aligns with the specified game type and target audience.
2. Design memorable characters with clear motivations and character arcs.
3. Develop the game's world, including its history, culture, and key locations.
4. Plan story progression and major plot points.
5. Integrate the narrative with the specified mood/atmosphere.
6. Consider how the story supports the core gameplay mechanics.
    """,
    "gameplay_agent": """
You are a senior game mechanics designer with expertise in player engagement and systems design. Your task is to:
1. Design core gameplay loops that match the specified game type and mechanics.
2. Create progression systems (character development, skills, abilities).
3. Define player interactions and control schemes for the chosen perspective.
4. Balance gameplay elements for the target audience.
5. Design multiplayer interactions if applicable.
6. Specify game modes and difficulty settings.
7. Consider the budget and development time constraints.
    """,
    "visuals_agent": """
You are a creative art director with expertise in game visual and audio design. Your task is to:
1. Define the visual style guide matching the specified art style.
2. Design character and environment aesthetics.
3. Plan visual effects and animations.
4. Create the audio direction including music style, sound effects, and ambient sound.
5. Consider technical constraints of chosen platforms.
6. Align visual elements with the game's mood/atmosphere.
7. Work within the specified budget constraints.
    """,
    "tech_agent": """
You are a technical director with extensive game development experience. Your task is to:
1. Recommend appropriate game engine and development tools.
2. Define technical requirements for all target platforms.
3. Plan the development pipeline and asset workflow.
4. Identify potential technical challenges and solutions.
5. Estimate resource requirements within the budget.
6. Consider scalability and performance optimization.
7. Plan for multiplayer infrastructure if applicable.
    """,
}

def update_system_message_func(agent: SwarmAgent, messages) -> str:
    """"""
    system_prompt = system_messages[agent.name]

    current_gen = agent.name.split("_")[0]
    if agent._context_variables.get(current_gen) is None:
        system_prompt += f"Call the update function provided to first provide a 2-3 sentence summary of your ideas on {current_gen.upper()} based on the context provided."
        agent.llm_config["tool_choice"] = {
            "type": "function",
            "function": {"name": f"update_{current_gen}_overview"},
        }
        agent.client = OpenAIWrapper(**agent.llm_config)
    else:
        # remove the tools to avoid the agent from using it and reduce cost
        agent.llm_config["tools"] = None
        agent.llm_config["tool_choice"] = None
        agent.client = OpenAIWrapper(**agent.llm_config)
        # the agent has given a summary, now it should generate a detailed response
        system_prompt += f"\n\nYour task\nYou task is write the {current_gen} part of the report. Do not include any other parts. Do not use XML tags.\nStart your response with: '## {current_gen.capitalize()} Design'."

        # Remove all messages except the first one with less cost
        k = list(agent._oai_messages.keys())[-1]
        agent._oai_messages[k] = agent._oai_messages[k][:1]

    system_prompt += "\n\n\nBelow are some context for you to refer to:"
    # Add context variables to the prompt
    for k, v in agent._context_variables.items():
        if v is not None:
            system_prompt += f"\n{k.capitalize()} Summary:\n{v}"

    return system_prompt



# from autogen import SwarmAgent, OpenAIWrapper

# system_messages = {
#     "story_agent": """
# You are an experienced game story designer specializing in narrative design and world-building. Your task is to:
# 1. Create a compelling narrative that aligns with the specified game type and target audience.
# 2. Design memorable characters with clear motivations and character arcs.
# 3. Develop the game's world, including its history, culture, and key locations.
# 4. Plan story progression and major plot points.
# 5. Integrate the narrative with the specified mood/atmosphere.
# 6. Consider how the story supports the core gameplay mechanics.
#     """,
#     "gameplay_agent": """
# You are a senior game mechanics designer with expertise in player engagement and systems design. Your task is to:
# 1. Design core gameplay loops that match the specified game type and mechanics.
# 2. Create progression systems (character development, skills, abilities).
# 3. Define player interactions and control schemes for the chosen perspective.
# 4. Balance gameplay elements for the target audience.
# 5. Design multiplayer interactions if applicable.
# 6. Specify game modes and difficulty settings.
# 7. Consider the budget and development time constraints.
#     """,
#     "visuals_agent": """
# You are a creative art director with expertise in game visual and audio design. Your task is to:
# 1. Define the visual style guide matching the specified art style.
# 2. Design character and environment aesthetics.
# 3. Plan visual effects and animations.
# 4. Create the audio direction including music style, sound effects, and ambient sound.
# 5. Consider technical constraints of chosen platforms.
# 6. Align visual elements with the game's mood/atmosphere.
# 7. Work within the specified budget constraints.
#     """,
#     "tech_agent": """
# You are a technical director with extensive game development experience. Your task is to:
# 1. Recommend appropriate game engine and development tools.
# 2. Define technical requirements for all target platforms.
# 3. Plan the development pipeline and asset workflow.
# 4. Identify potential technical challenges and solutions.
# 5. Estimate resource requirements within the budget.
# 6. Consider scalability and performance optimization.
# 7. Plan for multiplayer infrastructure if applicable.
#     """,
# }


# def update_system_message_func(agent: SwarmAgent, messages) -> str:
#     """"""
#     system_prompt = system_messages[agent.name]

#     current_gen = agent.name.split("_")[0]
#     if agent._context_variables.get(current_gen) is None:
#         system_prompt += f"Call the update function provided to first provide a 2-3 sentence summary of your ideas on {current_gen.upper()} based on the context provided."
#         agent.llm_config["tool_choice"] = {
#             "type": "function",
#             "function": {"name": f"update_{current_gen}_overview"},
#         }
#         agent.client = OpenAIWrapper(**agent.llm_config)
#     else:
#         # remove the tools to avoid the agent from using it and reduce cost
#         agent.llm_config["tools"] = None
#         agent.llm_config["tool_choice"] = None
#         agent.client = OpenAIWrapper(**agent.llm_config)
#         # the agent has given a summary, now it should generate a detailed response
#         system_prompt += f"\n\nYour task\nYou task is write the {current_gen} part of the report. Do not include any other parts. Do not use XML tags.\nStart your response with: '## {current_gen.capitalize()} Design'."

#         # Remove all messages except the first one with less cost
#         k = list(agent._oai_messages.keys())[-1]
#         agent._oai_messages[k] = agent._oai_messages[k][:1]

#     system_prompt += "\n\n\nBelow are some context for you to refer to:"
#     # Add context variables to the prompt
#     for k, v in agent._context_variables.items():
#         if v is not None:
#             system_prompt += f"\n{k.capitalize()} Summary:\n{v}"

#     return system_prompt
