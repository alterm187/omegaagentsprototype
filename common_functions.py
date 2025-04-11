import logging
import time
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union

from LLMConfiguration import LLMConfiguration, logger

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
         # Handle cases where LLM config might be missing/invalid, e.g., API key issues
         logger.error(f"LLM configuration invalid or missing for agent {name}. Cannot create agent.")
         # Depending on desired robustness, you might raise an error or return a placeholder
         # For now, let's raise an error to make the issue clear during setup
         raise ValueError(f"Invalid LLM configuration for agent {name}")

    if agent_type == "user_proxy":
        # UI will handle input, so set human_input_mode to NEVER
        agent = UserProxyAgent(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER", # Changed from ALWAYS
            code_execution_config=False, # Assuming no code execution needed by proxy
            llm_config=config,
            # Default auto reply can be empty if UI handles all interaction logic
            default_auto_reply="",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )
    else:  # AssistantAgent
        agent = AssistantAgent(
            name=name,
            system_message=system_message,
            llm_config=config,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            )
    return agent


# --- Custom Speaker Selection Logic (Keep as is for now) ---
def custom_speaker_selection(
        last_speaker: Agent,
        groupchat: GroupChat
) -> Agent:
    # If no messages, default to Boss (UserProxyAgent)
    boss_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not boss_agent:
        # This should ideally not happen if setup is correct
        logger.error("No UserProxyAgent found in custom_speaker_selection!")
        # Fallback to the first agent if Boss isn't found
        return groupchat.agents[0]

    if not groupchat.messages:
        logger.info("No messages, defaulting to Boss")
        return boss_agent

    # Get the last message
    try:
        last_message = groupchat.messages[-1]
        message_content = last_message.get('content', '')
        if not isinstance(message_content, str):
            # Handle non-string content if necessary, or default
            message_content = str(message_content) # Attempt conversion

        # Check for termination
        if message_content.rstrip().endswith("TERMINATE"):
             # If the last message is a termination message, maybe no one should speak next?
             # Or maybe the Boss to acknowledge? For now, let's just return None or Boss.
             logger.info("Termination message detected.")
             # Returning None might break manager loops, let's return Boss for potential wrap-up.
             # return None
             return boss_agent # Let Boss potentially acknowledge or end.

    except (AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Error accessing last message content ({e}), defaulting to Boss")
        return boss_agent

    # --- Agent Mention Logic (Simplified Example - Adapt patterns as needed) ---
    # This part is highly specific to your agent names and how they might call each other.
    # Keep the specific patterns from your original code if they are crucial.
    agent_patterns = {
         "PolicyGuard": ["PolicyGuard"],
         "FirstLineChallenger": ["FirstLineChallenger"],
         # Add other agents if needed
     }

    # Check for mentions (case-insensitive for robustness)
    lower_message_content = message_content.lower()
    for agent in groupchat.agents:
         if agent.name in agent_patterns:
             patterns = agent_patterns[agent.name]
             for pattern in patterns:
                 if pattern.lower() in lower_message_content:
                     logger.info(f"Pattern match found for {agent.name}, selecting.")
                     # time.sleep(4) # Consider removing long sleeps for interactive UI
                     return agent

    # --- Default Logic ---
    # If the last speaker was an assistant, often the user (Boss) should speak next.
    # If the last speaker was the user (Boss), the system should decide the next assistant.
    # AutoGen's default (`ROUND_ROBIN` or `auto`) might be simpler here unless complex routing is essential.
    # Let's try defaulting back to the Boss if no specific mention is found,
    # assuming the assistants address the user if unsure.
    logger.info(f"No specific pattern found or last speaker was {last_speaker.name}. Defaulting to Boss.")
    return boss_agent


def create_groupchat(agents: Sequence[Agent], max_round: int = 50) -> GroupChat:
     """Creates a GroupChat object."""
     if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent (like 'Boss').")

     return GroupChat(
         agents=list(agents),
         messages=[],
         max_round=max_round,
         # Use the custom speaker selection logic
         speaker_selection_method=custom_speaker_selection,
         allow_repeat_speaker=False, # Or True if agents need consecutive turns
     )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    """Creates a GroupChatManager."""
    config = llm_config.get_config()
    if not config:
        raise ValueError("Invalid LLM configuration for GroupChatManager.")

    return GroupChatManager(
        groupchat=groupchat,
        llm_config=config,
         # Add termination message check if manager should also recognize it
         is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )


def initiate_chat_task(
    user_agent: UserProxyAgent,
    manager: GroupChatManager,
    initial_prompt: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Initiates the chat by sending the first message from the user agent.
    Returns the initial message list and the next speaker determined by the selection method.
    """
    # Clear previous messages and add the initial prompt as if spoken by the user_agent
    manager.groupchat.reset() # Clear messages
    initial_message = {
        "role": "user", # AutoGen expects 'user' role for initiator usually
        "content": initial_prompt,
        "name": user_agent.name # Associate message with the boss/user agent
    }
    # Manually add the first message to the history
    manager.groupchat.messages.append(initial_message)

    # Determine the very first speaker *after* the initial prompt
    # Pass the user_agent as the 'last_speaker' to the selection function
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)

    logger.info(f"Chat initiated. First message sent by {user_agent.name}. Next speaker: {next_speaker.name if next_speaker else 'None'}")
    return manager.groupchat.messages, next_speaker # Return history and next agent


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
        # `generate_reply` is the core function for an agent to produce a response
        # It internally calls the LLM and adds the response to the chat history.
        # We might need to call manager._process_received_message or similar
        # depending on how GroupChatManager orchestrates turns.

        # Let's try using the manager's run_chat logic but limiting it.
        # This is tricky, as run_chat usually runs the whole loop.
        # Alternative: Manually trigger speaker's reply and then select next.

        # 1. Get reply from the current speaker
        reply = speaker.generate_reply(messages=manager.groupchat.messages, sender=manager)

        # 2. Process the reply (add to history, etc.) - speaker.send might do this
        if reply is not None:
            # Ensure reply is dict for consistency if needed, though send handles str
            if isinstance(reply, str):
                 processed_reply = {"role": "assistant", "content": reply, "name": speaker.name}
            elif isinstance(reply, dict):
                 processed_reply = reply
                 # Ensure 'name' field is present if using custom selection based on it
                 if 'name' not in processed_reply:
                      processed_reply['name'] = speaker.name
            else:
                 # Handle unexpected reply format
                 logger.warning(f"Unexpected reply format from {speaker.name}: {type(reply)}")
                 processed_reply = {"role": "assistant", "content": str(reply), "name": speaker.name}


            # Use manager.send to ensure message is added correctly to history
            # Need a recipient - often the manager itself or the previous speaker
            # Let's send it back to the manager to update the groupchat state.
            # Note: This might implicitly add the message to groupchat.messages via hooks.
            # If `speaker.send` already adds to `manager.groupchat.messages`,
            # we might double-add. We need to check AutoGen's internals or test carefully.
            # Let's assume `generate_reply` gives the content, and we manage history adding.

            # Check if generate_reply already added the message
            message_already_added = False
            if manager.groupchat.messages and manager.groupchat.messages[-1].get("content") == processed_reply.get("content") and manager.groupchat.messages[-1].get("name") == speaker.name:
                 message_already_added = True
                 new_messages.append(manager.groupchat.messages[-1]) # Get the actual message dict added
                 logger.info(f"Message from {speaker.name} seems already added by generate_reply.")

            if not message_already_added:
                 # Manually add the message if generate_reply didn't
                 manager.groupchat.messages.append(processed_reply)
                 new_messages.append(processed_reply)
                 logger.info(f"Manually added message from {speaker.name} to history.")


        # 3. Determine the next speaker *after* the current agent's turn
        if manager.groupchat.messages: # Ensure there are messages to base selection on
            next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
            logger.info(f"Step completed for {speaker.name}. Next speaker: {next_speaker.name if next_speaker else 'None'}")
        else:
            logger.warning("No messages in groupchat after step, cannot select next speaker.")
            # Decide fallback: maybe Boss?
            next_speaker = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), manager.groupchat.agents[0])


    except Exception as e:
        logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        # Decide how to handle errors: stop, default to Boss, etc.
        # Defaulting to Boss might allow user intervention.
        next_speaker = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)

    # Return only the messages generated in *this* step and the next speaker
    return new_messages, next_speaker

def send_user_message(
    manager: GroupChatManager,
    user_agent: UserProxyAgent,
    user_message: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Sends a message from the user (via user_agent) into the chat.
    Returns the message added and the next speaker.
    """
    if not user_message:
        logger.warning("Empty user message received.")
        # Decide if this should proceed or just return current state
        # Returning current state seems safest. We need the current speaker.
        # This requires passing the 'current_speaker' or getting it.
        # Let's assume this function is called ONLY when it's the user's turn.
        # The 'next_speaker' would be determined after the user speaks.

        # Re-select speaker based on last *actual* message before empty user input
        last_speaker = manager.groupchat.agents_by_name[manager.groupchat.messages[-1]['name']] if manager.groupchat.messages else user_agent
        next_speaker = manager.groupchat.select_speaker(last_speaker, manager.groupchat)
        return [], next_speaker # Return no new message, but the potentially recalculated next speaker


    # Construct the message dictionary
    message_dict = {
        "role": "user", # Role should match what the agents expect from the user proxy
        "content": user_message.strip(),
        "name": user_agent.name
    }

    # Add the user's message to the chat history
    manager.groupchat.messages.append(message_dict)
    logger.info(f"User message from {user_agent.name} added to history.")

    # Determine the next speaker *after* the user has spoken
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    logger.info(f"User message sent. Next speaker: {next_speaker.name if next_speaker else 'None'}")

    # Return the message that was just added and the next speaker
    return [message_dict], next_speaker
