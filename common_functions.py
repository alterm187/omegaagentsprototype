import logging
import time # Keep import if needed elsewhere, but sleep removed from selection
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union

from LLMConfiguration import LLMConfiguration, logger # Assuming logger is correctly configured here

# Configure logging (if not already done elsewhere)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_system_message(filename):
    """Reads system message from a file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"System message file not found: {filename}")
        return "You are a helpful assistant." # Default fallback
    except Exception as e:
        logger.error(f"Error reading system message file {filename}: {e}")
        return "You are a helpful assistant."


def create_agent(name: str, system_message_file: str, llm_config: LLMConfiguration, agent_type="assistant") -> autogen.Agent:
    """Creates an agent with the specified configuration."""
    system_message = read_system_message(system_message_file)
    config = llm_config.get_config()
    if not config:
         logger.error(f"LLM configuration invalid or missing for agent {name}. Cannot create agent.")
         raise ValueError(f"Invalid LLM configuration for agent {name}")

    if agent_type == "user_proxy":
        # UI will handle input, so set human_input_mode to NEVER
        agent = UserProxyAgent(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER", # Correct for UI-driven input
            code_execution_config=False,
            llm_config=config,
            default_auto_reply="", # UI handles replies
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )
    else:  # AssistantAgent
        agent = AssistantAgent(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER", # Ensure assistant agents don't ask for input
            llm_config=config,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            )
    return agent


# --- Updated Custom Speaker Selection Logic ---
def custom_speaker_selection(last_speaker: Agent, groupchat: GroupChat) -> Agent:
    """
    Selects the next speaker based on mentions in the last message.
    Defaults to Boss if no specific mention is found or on error.
    """
    boss_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not boss_agent:
        logger.error("No UserProxyAgent (Boss) found in custom_speaker_selection!")
        # Fallback to the first agent in the list if Boss isn't found (should not happen)
        return groupchat.agents[0]

    # Handle initial state or errors retrieving message
    if not groupchat.messages:
        logger.info("No messages yet, selecting Boss by default.")
        return boss_agent

    try:
        last_message = groupchat.messages[-1]
        message_content = last_message.get('content', '')
        if not isinstance(message_content, str):
            # Attempt conversion for safety, though content should usually be string
            message_content = str(message_content)

        # Prioritize TERMINATE check
        if message_content.rstrip().endswith("TERMINATE"):
            logger.info("Termination message detected. Selecting Boss for final step.")
            return boss_agent # Let Boss handle the termination state

    except (AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Error accessing last message content ({e}), defaulting to Boss")
        return boss_agent

    # --- Agent Mention Logic ---
    # Define agent names and simple patterns (agent name itself)
    agent_patterns = {
        "PolicyGuard": ["PolicyGuard"],
        "FirstLineChallenger": ["FirstLineChallenger"],
        "Boss": ["Boss"] # Allow agents to explicitly call the Boss
    }

    # Check for mentions (case-insensitive)
    lower_message_content = message_content.lower()
    mentioned_agents = []

    # Iterate through agents in the chat to find mentions
    for agent in groupchat.agents:
        if agent.name in agent_patterns:
            patterns = agent_patterns[agent.name]
            for pattern in patterns:
                # Check if the pattern (agent name) is in the message content
                if pattern.lower() in lower_message_content:
                    # Avoid selecting the same speaker immediately unless specifically allowed (which it isn't by default)
                    # Check groupchat settings `allow_repeat_speaker` if needed
                    # We'll just add the agent if mentioned.
                    mentioned_agents.append(agent)
                    # Break inner loop once a pattern for this agent is found
                    break

    # Filter out the last speaker from mentioned agents if allow_repeat_speaker is False (default)
    # Let's assume default Autogen behavior handles this implicitly or we handle it here.
    # A simple approach: pick the first mentioned agent that wasn't the last speaker.
    next_speaker = None
    for agent in mentioned_agents:
        if agent != last_speaker:
            next_speaker = agent
            break # Select the first valid mentioned agent

    if next_speaker:
        logger.info(f"Mention match found for '{next_speaker.name}'. Selecting.")
        return next_speaker

    # --- Default Logic ---
    # If no agent was mentioned or only the last speaker was mentioned, default to Boss.
    # This ensures the conversation flow returns to the user if the agents don't specify the next step.
    logger.info(f"No specific next agent mentioned (or only self-mention). Defaulting to Boss.")
    return boss_agent


def create_groupchat(agents: Sequence[Agent], max_round: int = 50) -> GroupChat:
     """Creates a GroupChat object using the custom speaker selection."""
     if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent (like 'Boss').")

     return GroupChat(
         agents=list(agents),
         messages=[],
         max_round=max_round,
         # Use the updated custom speaker selection logic
         speaker_selection_method=custom_speaker_selection,
         allow_repeat_speaker=False, # Standard setting
     )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    """Creates a GroupChatManager."""
    config = llm_config.get_config()
    if not config:
        raise ValueError("Invalid LLM configuration for GroupChatManager.")

    return GroupChatManager(
        groupchat=groupchat,
        llm_config=config, # Used for orchestrating if needed, e.g., summarizing
         is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )


# --- Updated Chat Initiation ---
def initiate_chat_task(
    user_agent: UserProxyAgent, # This is the 'Boss' agent
    manager: GroupChatManager,
    initial_prompt: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Initiates the chat by sending the first message from the user agent (Boss).
    Returns the initial message list and explicitly sets PolicyGuard as the next speaker.
    """
    manager.groupchat.reset() # Clear previous messages

    # Construct the initial message from the Boss/User
    initial_message = {
        "role": "user", # Consistent role for UserProxyAgent messages
        "content": initial_prompt.strip(),
        "name": user_agent.name
    }
    # Add the Boss's initial message to the history
    manager.groupchat.messages.append(initial_message)
    logger.info(f"Chat initiated by {user_agent.name}.")

    # Explicitly find and set PolicyGuard as the first agent to respond
    policy_guard_agent = manager.groupchat.agent_by_name("PolicyGuard")
    if not policy_guard_agent:
        logger.error("PolicyGuard agent not found in groupchat! Cannot set as first speaker.")
        # Fallback: Use the selection method to pick someone (might default back to Boss)
        next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
        logger.warning(f"Falling back to default speaker selection. Next: {next_speaker.name if next_speaker else 'None'}")
    else:
        next_speaker = policy_guard_agent
        logger.info(f"Explicitly selected 'PolicyGuard' as the first agent to speak.")

    # Return the history (containing only the initial message) and the determined next speaker
    return manager.groupchat.messages, next_speaker


# --- Agent Step Execution (No changes needed here based on discussion) ---
def run_agent_step(
    manager: GroupChatManager,
    speaker: Agent
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Runs a single step of the conversation for the given speaker.
    Returns the new messages generated in this step and the next speaker.
    """
    new_messages = []
    next_speaker = None
    try:
        # 1. Get reply from the current speaker
        # generate_reply expects list of messages, sender (manager acts as sender context)
        reply = speaker.generate_reply(messages=manager.groupchat.messages, sender=manager)

        # 2. Process the reply
        if reply is not None:
            # Ensure reply is a dictionary for consistent processing downstream if needed
            # Although GroupChat usually handles string replies internally when adding.
            if isinstance(reply, str):
                 processed_reply = {"role": "assistant", "content": reply, "name": speaker.name} # AutoGen convention
            elif isinstance(reply, dict):
                 processed_reply = reply
                 # Ensure 'name' field is present if using custom selection based on it
                 if 'name' not in processed_reply:
                      processed_reply['name'] = speaker.name
            else:
                 logger.warning(f"Unexpected reply format from {speaker.name}: {type(reply)}. Converting to string.")
                 processed_reply = {"role": "assistant", "content": str(reply), "name": speaker.name}

            # Check if autogen's generate_reply or internal hooks already added the message
            # This check might be fragile depending on AutoGen version/implementation details.
            message_already_added = False
            if manager.groupchat.messages and \
               manager.groupchat.messages[-1].get("content") == processed_reply.get("content") and \
               manager.groupchat.messages[-1].get("name") == speaker.name:
                 message_already_added = True
                 # If added, grab the actual dict from history to return it
                 new_messages.append(manager.groupchat.messages[-1])
                 logger.debug(f"Message from {speaker.name} appears added by generate_reply/hook.")

            if not message_already_added:
                 # Manually add the message if it wasn't automatically added
                 # Note: speaker.send(message, recipient) is another way, but adds complexity here.
                 # Directly appending to groupchat.messages is common in manual control loops.
                 manager.groupchat.messages.append(processed_reply)
                 new_messages.append(processed_reply)
                 logger.debug(f"Manually added message from {speaker.name} to history.")

        else:
            # Agent generated None reply (maybe termination or error within agent)
            logger.info(f"Agent {speaker.name} generated a None reply.")
            # Keep new_messages empty

        # 3. Determine the next speaker *after* the current agent's turn
        # Crucially, pass the *current speaker* as the 'last_speaker' context
        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        logger.info(f"Step completed for {speaker.name}. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")


    except Exception as e:
        # Log the error traceback for debugging
        logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        # Handle errors: Stop the chat, default to Boss, or try to recover?
        # Defaulting to Boss allows user intervention.
        next_speaker = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)
        logger.info(f"Error occurred. Defaulting next speaker to Boss: {next_speaker.name if next_speaker else 'None'}")
        # Optionally add an error message to the chat history?
        # error_message = {"role": "system", "content": f"Error during {speaker.name}'s turn: {e}", "name": "System"}
        # manager.groupchat.messages.append(error_message)
        # new_messages.append(error_message)


    # Return only the messages generated/added *in this step* and the next speaker
    return new_messages, next_speaker


# --- User Message Sending (No changes needed here based on discussion) ---
def send_user_message(
    manager: GroupChatManager,
    user_agent: UserProxyAgent, # Boss agent
    user_message: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Sends a message from the user (via user_agent/Boss) into the chat.
    Returns the message added and the next speaker determined by the selection method.
    """
    if not user_message or not user_message.strip():
        logger.warning("Empty user message received. Ignoring.")
        # If the user sends empty input, don't add it, and just re-select the speaker.
        # The last speaker was technically the Boss (UI), but the last *message* was from someone else.
        # Re-run selection based on the state *before* the empty input attempt.
        last_actual_speaker = manager.groupchat.agent_by_name(manager.groupchat.messages[-1]['name']) if manager.groupchat.messages else user_agent
        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        return [], next_speaker # Return no new message, just the recalculated next speaker

    # Construct the message dictionary
    message_dict = {
        "role": "user", # Role for UserProxyAgent
        "content": user_message.strip(),
        "name": user_agent.name # Name of the Boss agent
    }

    # Add the user's message to the chat history
    manager.groupchat.messages.append(message_dict)
    logger.info(f"User message from {user_agent.name} added to history.")

    # Determine the next speaker *after* the user (Boss) has spoken
    # Pass the user_agent as the 'last_speaker' context
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    logger.info(f"User message sent. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    # Return the message that was just added and the determined next speaker
    return [message_dict], next_speaker
