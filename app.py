import streamlit as st
import logging
from typing import Optional, Dict, List
import os
import autogen # To access Agent type hint if needed

# Import functions from our refactored modules
# (usually means they are in the same directory or the path is configured)
try:
    from main import setup_chat, BOSS_NAME # Import setup function and Boss agent name
    from common_functions import (
        initiate_chat_task,
        run_agent_step,
        send_user_message
    ) 
    
except ImportError as e:
    st.error(f"Failed to import necessary functions. Make sure main.py and common_functions.py are accessible. Error: {e}")
    st.stop() # Stop execution if imports fail

# Configure logging (optional, but helpful for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide") # Use wider layout

# Helper function to read system message from file
def _read_system_message(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        script_dir = os.path.dirname(__file__)
        alt_path = os.path.join(script_dir, file_path)
        with open(alt_path, "r") as f:
            return f.read()

st.title("AutoGen Group Chat Interface")

# --- Session State Initialization ---
# Initialize keys in session state if they don't exist
default_values = {
    "chat_initialized": False,
    "processing": False, # Flag to prevent double clicks
    "manager": None,
    "boss_agent": None,
    "messages": [], # List to store chat history {role/name: ..., content: ...}
    "next_agent": None, # Stores the Agent object whose turn it is
    "error_message": None,
    "initial_task_desc": "Describe the product or task here...",
    "initial_policy": "Provide policy content here..."
    ,"policy_guard_sys_msg": _read_system_message("PolicyGuard.md"), 
    "first_line_challenger_sys_msg": _read_system_message("FirstLineChallenger.md")
    }
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Helper Function to Display Messages ---
def display_messages(messages: List[Dict]):
    """Displays chat messages using Streamlit's chat elements."""
    for msg in messages:
        role = msg.get("name", msg.get("role", "Unknown")) # Use name if available
        content = msg.get("content", str(msg)) # Ensure content is string
        # Handle potential complex content (like tool calls) - display simply for now
        if not isinstance(content, str):
            content = str(content)
        # Use name for chat display role
        with st.chat_message(name=role):
            st.markdown(content) # Use markdown for potential formatting


# --- Sidebar for Inputs and Control ---
with st.sidebar:
    st.header("Configuration")

    # Use text_area for potentially longer inputs
    task_desc = st.text_area(
        "Task Description:",
        value=st.session_state.initial_task_desc,
        height=150,
        key="task_input",
        disabled=st.session_state.chat_initialized or st.session_state.processing
    )
    policy_content = st.text_area(
        "Policy Content:",
        value=st.session_state.initial_policy,
        height=200,
        key="policy_input",
        disabled=st.session_state.chat_initialized or st.session_state.processing
    )
    st.header("Agent System Messages")
    
    policy_guard_sys_msg = st.text_area(
        "PolicyGuard System Message:",
        value=st.session_state.get("policy_guard_sys_msg", ""),  # Retrieve from state or default
        height=200,
        key="policy_guard_sys_msg_input",
        disabled=st.session_state.chat_initialized or st.session_state.processing,
    )

    first_line_challenger_sys_msg = st.text_area(
        "FirstLineChallenger System Message:",
        value=st.session_state.get("first_line_challenger_sys_msg", ""),  # Retrieve from state or default
        height=200,
        key="first_line_challenger_sys_msg_input",
        disabled=st.session_state.chat_initialized or st.session_state.processing,
    )
    
    if st.button("💾 Save System Messages", disabled=st.session_state.chat_initialized or st.session_state.processing, key="save_sys_msgs"):
        with st.spinner("Saving system messages..."):
            # Save system messages to files using tool.
            default_api.natural_language_write_file(path="PolicyGuard.md", prompt=policy_guard_sys_msg, language="markdown")
            default_api.natural_language_write_file(path="FirstLineChallenger.md", prompt=first_line_challenger_sys_msg, language="markdown")
        st.success("System messages saved!")

    start_button_disabled = not task_desc or not policy_content or st.session_state.chat_initialized or st.session_state.processing
    if st.button("🚀 Start Chat", key="start_button", disabled=start_button_disabled):
        st.session_state.processing = True
        st.session_state.error_message = None # Clear previous errors
        with st.spinner("Setting up agents and initiating chat..."):
            try:
                # Before calling setup_chat, update session state with current system messages
                st.session_state.policy_guard_sys_msg = policy_guard_sys_msg 
                st.session_state.first_line_challenger_sys_msg = first_line_challenger_sys_msg 
            

                # 1. Setup Agents and Manager 
                st.session_state.manager, st.session_state.boss_agent = setup_chat() # Using defaults for now

                # 2. Prepare Initial Prompt (Combine task and policy)
                # How policy is injected depends on agent design. Assume PolicyGuard expects it in the initial message.
                initial_prompt = f"**Task Description:**\n{task_desc}\n\n**Policy Content:**\n{policy_content}\n\nPlease analyze according to the policy."
                logger.info(f"Initiating chat with prompt:\n{initial_prompt}")

                # 3. Initiate Chat Task
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.boss_agent,
                    st.session_state.manager,
                    initial_prompt
                )

                # 4. Update State
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True
                logger.info(f"Chat initialized. First message sent. Next agent: {next_agent.name if next_agent else 'None'}")

            except (FileNotFoundError, ValueError, KeyError, ImportError, AttributeError, Exception) as e:
                logger.error(f"Error during chat setup or initiation: {e}", exc_info=True)
                st.session_state.error_message = f"Error: {e}"
                st.session_state.chat_initialized = False # Ensure state reflects failure
            finally:
                st.session_state.processing = False
                st.rerun() # Rerun to update UI based on new state
    

# --- Main Chat Area ---
st.header("Group Chat")

# Display Error Messages
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# Display Chat History if Initialized
if st.session_state.chat_initialized:
    display_messages(st.session_state.messages)

    # --- Interaction Controls ---
    if st.session_state.next_agent and not st.session_state.processing:
        next_agent_name = st.session_state.next_agent.name

        # Check if it's the User's (Boss's) turn
        if next_agent_name == BOSS_NAME:
            st.markdown(f"**Your turn (as {BOSS_NAME}):**")
            user_input = st.text_input("Enter your message:", key=f"user_input_{len(st.session_state.messages)}", disabled=st.session_state.processing)
            if st.button("✉️ Send Message", key=f"send_{len(st.session_state.messages)}", disabled=st.session_state.processing or not user_input):
                st.session_state.processing = True
                st.session_state.error_message = None
                with st.spinner(f"Sending message as {BOSS_NAME}..."):
                    try:
                        new_messages, next_agent = send_user_message(
                            st.session_state.manager,
                            st.session_state.boss_agent,
                            user_input
                        )
                        st.session_state.messages.extend(new_messages)
                        st.session_state.next_agent = next_agent
                        logger.info(f"User message sent. Next agent: {next_agent.name if next_agent else 'None'}")
                    except Exception as e:
                        logger.error(f"Error sending user message: {e}", exc_info=True)
                        st.session_state.error_message = f"Error sending message: {e}"
                    finally:
                        st.session_state.processing = False
                        st.rerun()

        # Otherwise, it's an AI Agent's turn
        else:
            st.markdown(f"**Next turn:** {next_agent_name}")
            if st.button(f"▶️ Run {next_agent_name}'s Turn", key=f"run_agent_{len(st.session_state.messages)}", disabled=st.session_state.processing):
                st.session_state.processing = True
                st.session_state.error_message = None
                with st.spinner(f"Running {next_agent_name}'s turn..."):
                    try:
                        new_messages, next_agent = run_agent_step(
                            st.session_state.manager,
                            st.session_state.next_agent # Pass the actual agent object
                        )
                        st.session_state.messages.extend(new_messages)
                        st.session_state.next_agent = next_agent
                        logger.info(f"Agent {next_agent_name} finished. Next agent: {next_agent.name if next_agent else 'None'}")

                        # Check for termination message from the last agent
                        if new_messages and new_messages[-1].get("content", "").rstrip().endswith("TERMINATE"):
                             st.info(f"Agent {next_agent_name} terminated the chat.")
                             st.session_state.next_agent = None # Stop processing

                    except Exception as e:
                        logger.error(f"Error running agent step for {next_agent_name}: {e}", exc_info=True)
                        st.session_state.error_message = f"Error during {next_agent_name}'s turn: {e}"
                        # Decide if chat should stop on error, maybe set next_agent to None or Boss
                        st.session_state.next_agent = st.session_state.boss_agent # Allow user intervention on error?
                    finally:
                        st.session_state.processing = False
                        st.rerun()

    elif not st.session_state.next_agent and st.session_state.chat_initialized:
        # Chat has ended (either normally via TERMINATE or potentially an error)
        st.success("Chat has concluded or the next speaker could not be determined.")
        # Optionally add a button to reset/start a new chat
        if st.button("🔄 Start New Chat"):
             # Reset relevant state variables
             for key in default_values:
                  st.session_state[key] = default_values[key]
             # Need to also clear the input widgets if using 'key'
             st.session_state.task_input = default_values["initial_task_desc"]
             st.session_state.policy_input = default_values["initial_policy"]
             st.rerun()


elif not st.session_state.chat_initialized and not st.session_state.error_message:
    st.info("Please provide the Task Description and Policy Content in the sidebar, then click 'Start Chat'.")
