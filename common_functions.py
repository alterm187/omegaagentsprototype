# Corrected full content for common_functions.py v3

import logging
import time # Keep import if needed elsewhere, but sleep removed from selection
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union
import json # Import json for pretty printing dicts

from LLMConfiguration import LLMConfiguration, logger # Assuming logger is correctly configured here

# Configure logging (if not already done elsewhere)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # User should ensure DEBUG level is set in active config


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


# --- Updated Custom Speaker Selection Logic -- WITH DEBUG LOGGING ---
def custom_speaker_selection(last_speaker: Agent, groupchat: GroupChat) -> Agent:
    """
    Selects the next speaker based on mentions in the last message.
    Defaults to Boss if no specific mention is found or on error.
    """
    # DEBUG: Log entry point and arguments
    logger.debug(f"--- Entering custom_speaker_selection ---")
    logger.debug(f"Last speaker: {last_speaker.name if last_speaker else 'None'}")
    logger.debug(f"Available agents: {[a.name for a in groupchat.agents]}")

    boss_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not boss_agent:
        logger.error("No UserProxyAgent (Boss) found in custom_speaker_selection!")
        # Fallback to the first agent in the list if Boss isn't found (should not happen)
        logger.debug("Selecting first agent as fallback (Boss not found).")
        return groupchat.agents[0]

    # Handle initial state or errors retrieving message
    if not groupchat.messages:
        logger.info("No messages yet, selecting Boss by default.")
        logger.debug(f"--- Exiting custom_speaker_selection (No messages) ---")
        return boss_agent

    last_message = None
    message_content = ''
    try:
        last_message = groupchat.messages[-1]
        # DEBUG: Log the last message received
        try:
            logger.debug(f"Last message received by custom_speaker_selection: {json.dumps(last_message)}")
        except Exception as e: # Catch potential errors during logging the message itself
             logger.warning(f"Could not json.dumps last_message for logging in custom_speaker_selection: {e}")
             logger.debug(f"Last message (raw) received by custom_speaker_selection: {last_message}")


        message_content = last_message.get('content', '')
        if not isinstance(message_content, str):
            # Attempt conversion for safety, though content should usually be string
            logger.warning(f"Message content was not a string ({type(message_content)}), converting to string.")
            message_content = str(message_content)

        logger.debug(f"Extracted message content for selection check: '{message_content[:100]}...'") # Log first 100 chars

        # Prioritize TERMINATE check
        if message_content.rstrip().endswith("TERMINATE"):
            logger.info("Termination message detected. Selecting Boss for final step.")
            logger.debug(f"--- Exiting custom_speaker_selection (Terminate) ---")
            return boss_agent # Let Boss handle the termination state

    except (AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Error accessing last message content ({e}), defaulting to Boss")
        logger.debug(f"--- Exiting custom_speaker_selection (Error accessing message) ---")
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
                    logger.debug(f"Found potential mention of '{agent.name}' using pattern '{pattern}'.")
                    mentioned_agents.append(agent)
                    # Break inner loop once a pattern for this agent is found
                    break

    # Filter out the last speaker from mentioned agents if allow_repeat_speaker is False (default)
    next_speaker = None
    logger.debug(f"Mentioned agents (raw): {[a.name for a in mentioned_agents]}")
    for agent in mentioned_agents:
        if agent != last_speaker:
            next_speaker = agent
            logger.debug(f"Found valid mention for next speaker: {next_speaker.name}")
            break # Select the first valid mentioned agent

    if next_speaker:
        logger.info(f"Mention match found for '{next_speaker.name}'. Selecting.")
        logger.debug(f"--- Exiting custom_speaker_selection (Mention found) ---")
        return next_speaker

    # --- Default Logic ---
    logger.info(f"No specific next agent mentioned (or only self-mention). Defaulting to Boss.")
    logger.debug(f"--- Exiting custom_speaker_selection (Defaulting to Boss) ---")
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


# --- Updated Chat Initiation (Fixed - Adds initial message to history) ---
def initiate_chat_task(
    user_agent: UserProxyAgent, # This is the 'Boss' agent
    manager: GroupChatManager,
    initial_prompt: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Initiates the chat: resets history, adds the first message from Boss to the
    manager's groupchat history, and sets the first speaker.
    Returns the initial message list (for display) and the next speaker.
    """
    manager.groupchat.reset() # Ensure chat history is clear before starting
    logger.info("GroupChat history reset.")

    initial_message = {
        "role": "user",
        "content": initial_prompt.strip(),
        "name": user_agent.name
    }
    # *** Add the initial message to the manager's history ***
    manager.groupchat.messages.append(initial_message)
    logger.info(f"Initial message from {user_agent.name} added to manager history.")
    # DEBUG: Log message history after adding initial message
    try:
        logger.debug(f"History after initial message: {json.dumps(manager.groupchat.messages)}")
    except Exception as e:
        logger.warning(f"Could not log initial message history: {e}")


    policy_guard_agent = manager.groupchat.agent_by_name("PolicyGuard")
    if not policy_guard_agent:
        logger.error("PolicyGuard agent not found in groupchat! Cannot explicitly set as first speaker.")
        # Select speaker based on the now-added initial message
        # DEBUG: Log that we are about to select speaker after initial message
        logger.debug("PolicyGuard not found, selecting speaker based on initial message...")
        next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat) # Call selection
        logger.warning(f"Falling back to default speaker selection. Next: {next_speaker.name if next_speaker else 'None'}")
    else:
        next_speaker = policy_guard_agent
        logger.info(f"Explicitly selected 'PolicyGuard' as the first agent to speak.")

    # Return the initial message (for display) and the selected next speaker.
    # app.py will add this message to st.session_state.messages and then trigger the first turn.
    return [initial_message], next_speaker


# --- Agent Step Execution (Revised Logic - WITH DEBUG LOGGING) ---
def run_agent_step(
    manager: GroupChatManager,
    speaker: Agent
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Runs a single step of the conversation for the given speaker using revised logic.
    Returns ONLY the new messages generated/added in this step and the next speaker.
    It handles cases where autogen might or might not add the message to history.
    """
    newly_added_messages = []
    next_speaker = None
    try:
        logger.info(f"--- Running step for agent: {speaker.name} ---")
        # Use the *current* messages from the manager's groupchat as context
        messages_context = manager.groupchat.messages
        len_before_reply = len(messages_context)
        logger.info(f"Messages history length before {speaker.name}.generate_reply: {len_before_reply}")
        # DEBUG: Log last message before generation (if history exists)
        if messages_context:
             try:
                 logger.debug(f" Last message before generation: {json.dumps(messages_context[-1])}")
             except Exception as e:
                 logger.warning(f"Could not log last message before generation: {e}")
        else:
            logger.debug(" Message history is empty before generation.")


        # 1. Get reply from the current speaker.
        # Provide the manager as the sender context.
        reply = speaker.generate_reply(messages=messages_context, sender=manager)

        # Log the raw reply for diagnostics
        try:
            log_reply_content_str = json.dumps(reply, indent=2) if isinstance(reply, dict) else str(reply)
            logger.info(f"Raw reply generated by {speaker.name}: {log_reply_content_str[:500]}...")
        except Exception as log_e:
            logger.error(f"Error logging raw reply: {log_e}")

        # 2. Check if the message history was updated by the framework
        # Access the potentially updated list directly from the manager
        messages_after_reply = manager.groupchat.messages
        len_after_reply = len(messages_after_reply)
        num_new_messages = len_after_reply - len_before_reply
        logger.info(f"Messages history length after {speaker.name}.generate_reply: {len_after_reply} ({num_new_messages} new)")

        if num_new_messages > 0:
            # Framework added the message(s). Capture them from the end of the list.
            newly_added_messages = messages_after_reply[len_before_reply:]
            logger.info(f"Captured {num_new_messages} new message(s) automatically added by framework for {speaker.name}.")
            # DEBUG: Log auto-added messages
            for i, msg in enumerate(newly_added_messages):
               try:
                   logger.debug(f"  Auto-added msg {i+1}: {json.dumps(msg)}")
               except Exception: logger.debug(f"  Auto-added msg {i+1} (str): {msg}")

        elif reply is not None:
            # Framework did NOT add the message, but a reply exists. Manually add it.
            logger.warning(f"Agent {speaker.name} generated a reply but message count didn't increase. Manually adding.")
            reply_content = None
            # Extract content: Prefer dict['content'], fallback to string
            if isinstance(reply, dict):
                reply_content = reply.get("content")
            elif isinstance(reply, str):
                reply_content = reply
            else:
                logger.error(f"Agent {speaker.name} generated reply in unexpected format: {type(reply)}. Cannot add manually.")

            if reply_content is not None: # Check content exists, even if empty string
                # Construct the message dictionary
                # Determine role based on agent type (UserProxyAgent is 'user', others 'assistant')
                role = "user" if isinstance(speaker, UserProxyAgent) else "assistant"
                manual_message = {
                    "role": role,
                    "content": reply_content,
                    "name": speaker.name # Use the speaker's name
                }
                # Append to the *shared* message history
                # DEBUG: Log *before* manual append
                try:
                    logger.debug(f"Attempting to manually append message: {json.dumps(manual_message)}") # Added try-except
                except Exception as e:
                     logger.warning(f"Could not log manual message before append: {e}")


                manager.groupchat.messages.append(manual_message)
                # DEBUG: Log *after* manual append
                logger.debug(f"Manually appended. History length now: {len(manager.groupchat.messages)}")
                try:
                    logger.debug(f" Last message in history after manual append: {json.dumps(manager.groupchat.messages[-1])}")
                except Exception as e:
                    logger.warning(f"Could not log last message after manual append: {e}")

                newly_added_messages = [manual_message] # This is the message added in this step
                logger.info(f"Manually added message from {speaker.name} (Role: {role}) to history.")
            else:
                logger.warning(f"Could not extract valid content from the reply generated by {speaker.name} (reply_content is None). No message added manually.")
                newly_added_messages = [] # Ensure it's empty
        else:
             # Reply is None, and no message was added automatically.
             logger.info(f"Agent {speaker.name} generated no reply (reply is None) and no message was added automatically.")
             newly_added_messages = [] # Ensure it's empty

        # 3. Determine the next speaker using the updated history
        # Pass the current speaker who just finished as the context for selection.
        # DEBUG: Log just before calling select_speaker
        logger.debug(f"Calling manager.groupchat.select_speaker. Last speaker was {speaker.name}.")
        try:
            logger.debug(f" History state just before speaker selection: Last message: {json.dumps(manager.groupchat.messages[-1]) if manager.groupchat.messages else 'Empty'}")
        except Exception as e:
            logger.warning(f"Could not log history state before speaker selection: {e}")

        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        # DEBUG: Log the result of speaker selection
        logger.debug(f"select_speaker returned: {next_speaker.name if next_speaker else 'None'}")
        logger.info(f"Step completed for {speaker.name}. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    except Exception as e:
        logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        # Try to gracefully default to Boss on error
        try:
            boss_agent = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)
            if boss_agent:
                 next_speaker = boss_agent
                 logger.info(f"Error occurred. Defaulting next speaker to Boss: {next_speaker.name}")
            else:
                 logger.error("Boss agent not found! Cannot default next speaker on error.")
                 next_speaker = None
        except Exception as fallback_e:
            logger.error(f"Failed to fallback to Boss agent after error: {fallback_e}")
            next_speaker = None # Indicate failure to select next speaker

    # Return ONLY the messages identified or added in this step and the next speaker
    return newly_added_messages, next_speaker


# --- User Message Sending (Revised to align with run_agent_step/manual add - No changes needed here) ---
def send_user_message(
    manager: GroupChatManager,
    user_agent: UserProxyAgent, # Boss agent
    user_message: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Handles sending a message from the user (Boss) into the chat.
    Manually constructs and adds the message to the groupchat history.
    Returns the message added and the next speaker determined by the selection method.
    """
    if not user_message or not user_message.strip():
        logger.warning("Empty user message received. Attempting to re-select speaker without adding message.")
        # Determine last speaker based on message history or default to user_agent if history is empty
        last_actual_speaker = None
        if manager.groupchat.messages:
             try:
                 last_msg_name = manager.groupchat.messages[-1].get('name')
                 if last_msg_name:
                      last_actual_speaker = manager.groupchat.agent_by_name(last_msg_name)
             except Exception as e:
                  logger.warning(f"Could not determine last speaker from history: {e}")
        if not last_actual_speaker:
             last_actual_speaker = user_agent # Default to user agent if history empty or error

        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        logger.info(f"Empty message ignored. Recalculated next speaker: {next_speaker.name if next_speaker else 'None'}")
        return [], next_speaker # Return no new message, just the recalculated next speaker

    # Manually construct and add the user message
    message_dict = {
        "role": "user", # Role is 'user' for UserProxyAgent
        "content": user_message.strip(),
        "name": user_agent.name
    }
    # Append directly to the manager's message list
    # DEBUG: Log *before* user message append
    try:
        logger.debug(f"Attempting to manually append user message: {json.dumps(message_dict)}") # Added try-except
    except Exception as e:
         logger.warning(f"Could not log user message before append: {e}")


    manager.groupchat.messages.append(message_dict)
    # DEBUG: Log *after* user message append
    logger.debug(f"Manually appended user message. History length now: {len(manager.groupchat.messages)}")
    try:
        logger.debug(f" Last message in history after user append: {json.dumps(manager.groupchat.messages[-1])}")
    except Exception as e:
        logger.warning(f"Could not log last message after user append: {e}")

    logger.info(f"User message from {user_agent.name} manually added to history.")


    # Determine next speaker *after* the user message has been added
    # The user_agent (Boss) is the one who just spoke.
    # DEBUG: Log just before calling select_speaker after user message
    logger.debug(f"Calling manager.groupchat.select_speaker after user message. Last speaker was {user_agent.name}.")
    try:
        logger.debug(f" History state just before speaker selection: Last message: {json.dumps(manager.groupchat.messages[-1]) if manager.groupchat.messages else 'Empty'}")
    except Exception as e:
        logger.warning(f"Could not log history state before speaker selection: {e}")

    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    # DEBUG: Log the result of speaker selection
    logger.debug(f"select_speaker returned: {next_speaker.name if next_speaker else 'None'}")
    logger.info(f"User message sent. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    # Return the manually added message and the next speaker
    return [message_dict], next_speaker
