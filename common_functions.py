import logging
import time # Keep import if needed elsewhere, but sleep removed from selection
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union
import json # Import json for pretty printing dicts

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
                    mentioned_agents.append(agent)
                    # Break inner loop once a pattern for this agent is found
                    break

    # Filter out the last speaker from mentioned agents if allow_repeat_speaker is False (default)
    next_speaker = None
    for agent in mentioned_agents:
        if agent != last_speaker:
            next_speaker = agent
            break # Select the first valid mentioned agent

    if next_speaker:
        logger.info(f"Mention match found for '{next_speaker.name}'. Selecting.")
        return next_speaker

    # --- Default Logic ---
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
        llm_config=config,
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
    initial_message = {
        "role": "user",
        "content": initial_prompt.strip(),
        "name": user_agent.name
    }
    manager.groupchat.messages.append(initial_message)
    logger.info(f"Chat initiated by {user_agent.name}.")

    policy_guard_agent = manager.groupchat.agent_by_name("PolicyGuard")
    if not policy_guard_agent:
        logger.error("PolicyGuard agent not found in groupchat! Cannot set as first speaker.")
        next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
        logger.warning(f"Falling back to default speaker selection. Next: {next_speaker.name if next_speaker else 'None'}")
    else:
        next_speaker = policy_guard_agent
        logger.info(f"Explicitly selected 'PolicyGuard' as the first agent to speak.")

    return manager.groupchat.messages, next_speaker


# --- Agent Step Execution (FIXED message not being added) ---
def run_agent_step(
    manager: GroupChatManager,
    speaker: Agent
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Runs a single step of the conversation for the given speaker.
    Handles cases where generate_reply returns content but doesn't update history.
    Returns the new messages generated in this step and the next speaker.
    """
    newly_added_messages = [] # Initialize as empty list
    next_speaker = None
    try:
        # --- Start Pre-Generation Diagnostics ---
        logger.info(f"--- Running step for agent: {speaker.name} ---")
        messages_before_reply = list(manager.groupchat.messages) # Copy current messages
        len_before_reply = len(messages_before_reply)
        logger.info(f"Messages history length before {speaker.name}.generate_reply: {len_before_reply}")
        # --- End Pre-Generation Diagnostics ---

        # 1. Get reply from the current speaker.
        reply = speaker.generate_reply(messages=messages_before_reply, sender=manager)

        # --- Start Post-Generation Diagnostics ---
        try:
            logger.info(f"Raw reply potentially generated by {speaker.name}.generate_reply:")
            logger.info(f"  Type: {type(reply)}")
            log_reply_content = reply
            log_reply_content_str = ""
            try:
                log_reply_content_str = json.dumps(log_reply_content, indent=2)
            except TypeError:
                 log_reply_content_str = str(log_reply_content)
            logger.info(f"  Content: {log_reply_content_str[:500]}...")
        except Exception as log_e:
            logger.error(f"Error during post-generation logging: {log_e}")
        # --- End Post-Generation Diagnostics ---

        # 2. Process the reply (Check if message was added automatically or needs manual add)
        messages_after_reply = manager.groupchat.messages # Get potentially updated list
        len_after_reply = len(messages_after_reply)
        num_new_messages = len_after_reply - len_before_reply
        logger.info(f"Messages history length after {speaker.name}.generate_reply: {len_after_reply} ({num_new_messages} new)")

        if num_new_messages > 0:
            # Standard case: Agent/framework added the message(s) automatically
            newly_added_messages = messages_after_reply[len_before_reply:]
            logger.debug(f"Captured {num_new_messages} new message(s) added by {speaker.name}'s turn.")
        elif reply is not None:
            # --- Start Manual Add Logic ---
            logger.warning(f"Agent {speaker.name} generated a reply but message count didn't increase. Attempting to manually add.")
            reply_content = None
            # Extract content: Prefer dict['content'], fallback to string, else error
            if isinstance(reply, dict):
                reply_content = reply.get("content")
            elif isinstance(reply, str):
                reply_content = reply
            else:
                logger.error(f"Agent {speaker.name} generated reply in unexpected format: {type(reply)}. Cannot add manually.")

            if reply_content:
                # Construct the message dictionary (assuming speaker is assistant)
                manual_message = {
                    "role": "assistant", # Use 'assistant' role for agent messages
                    "content": reply_content,
                    "name": speaker.name # Use the speaker's name
                }
                # Append to the *shared* message history
                manager.groupchat.messages.append(manual_message)
                newly_added_messages = [manual_message] # This is the message added in this step
                logger.info(f"Manually added message from {speaker.name} to history.")
                # Update lengths for consistency if needed (optional)
                # len_after_reply = len(manager.groupchat.messages)
                # num_new_messages = len_after_reply - len_before_reply
            else:
                logger.warning(f"Could not extract valid content from the reply generated by {speaker.name}. No message added.")
                newly_added_messages = [] # Ensure it's empty if nothing was added
            # --- End Manual Add Logic ---
        else:
             # Handle the case where reply is None (e.g., agent decides not to speak)
             logger.info(f"Agent {speaker.name} generated no reply or message count didn't increase.")
             newly_added_messages = [] # Ensure it's empty


        # 3. Determine the next speaker (Use the potentially updated history)
        # Pass the current speaker (who just spoke) as the context for selection
        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        logger.info(f"Step completed for {speaker.name}. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    except Exception as e:
        logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        next_speaker = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)
        logger.info(f"Error occurred. Defaulting next speaker to Boss: {next_speaker.name if next_speaker else 'None'}")

    # Return ONLY the messages added in this step and the next speaker
    return newly_added_messages, next_speaker


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
        # Determine last speaker based on message history or default to user_agent if history is empty
        last_actual_speaker = manager.groupchat.agent_by_name(manager.groupchat.messages[-1]['name']) if manager.groupchat.messages else user_agent
        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        return [], next_speaker # Return no new message, just the recalculated next speaker

    message_dict = {
        "role": "user",
        "content": user_message.strip(),
        "name": user_agent.name
    }
    manager.groupchat.messages.append(message_dict)
    logger.info(f"User message from {user_agent.name} added to history.")

    # Determine next speaker after the user message
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    logger.info(f"User message sent. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    return [message_dict], next_speaker
