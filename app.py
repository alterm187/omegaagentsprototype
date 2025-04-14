import streamlit as st
import logging
import traceback
from main import (
    initialize_chat,
    run_agent_step,
    send_user_message,
    BOSS_NAME,
    load_config,
    config_path,
)
# Assuming other necessary imports are present

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Constants and Configuration ---
MAX_MESSAGES_DISPLAY = 50 # Limit messages displayed to prevent clutter

# --- Helper Functions ---
def display_messages(messages):
    """Displays chat messages, limiting the number shown."""
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY:
        st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    # Display relevant messages
    for i, msg in enumerate(messages[start_index:], start=start_index):
        sender_name = msg.get("name", "System") # Use 'name' if available, else check 'role'
        if not sender_name and "role" in msg:
             sender_name = msg["role"].capitalize() # Fallback to role if name isn't set

        content = msg.get("content", "")
        if isinstance(content, list): # Handle complex content (e.g., tool calls)
            content_str = ""
            for item in content:
                 if isinstance(item, dict) and "text" in item:
                      content_str += item["text"] + "
" # Fixed newline
                 elif isinstance(item, str): # Sometimes content is just a string in the list
                      content_str += item + "
" # Fixed newline
                 else: # Fallback for unexpected structure
                      content_str += str(item) + "
" # Fixed newline
            content = content_str.strip()
        elif not isinstance(content, str):
             content = str(content) # Ensure content is a string

        # Simple way to distinguish user (Boss) messages - adjust if needed
        if sender_name == BOSS_NAME:
            with st.chat_message("user", avatar="🧑‍💼"): # Or use a specific Boss avatar
                 st.markdown(f"**{sender_name}:**
{content}")
        else:
            # Simple heuristic to try and identify system/tool messages vs agent messages
            # This might need refinement based on actual message structure
            is_agent_message = "sender" in msg or ("role" in msg and msg["role"] not in ["system", "tool", "function"])

            if is_agent_message:
                 with st.chat_message("assistant", avatar="🤖"): # Generic AI avatar
                      st.markdown(f"**{sender_name}:**
{content}")
            else: # Likely a system message or tool call/result
                 with st.chat_message("system", avatar="⚙️"):
                      st.markdown(f"_{sender_name}: {content}_") # Italicize system/tool messages


# --- Streamlit App UI ---
st.title("🤖 Multi-Agent Chat")

# --- Initialization ---
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
    st.session_state.processing = False # Flag to prevent multiple simultaneous runs
    st.session_state.error_message = None
    st.session_state.config = None
    st.session_state.manager = None
    st.session_state.boss_agent = None
    st.session_state.messages = []
    st.session_state.next_agent = None
    st.session_state.initial_prompt = "" # Store initial prompt

# --- Configuration Loading ---
if not st.session_state.config:
    try:
        st.session_state.config = load_config(config_path)
        st.sidebar.success("Configuration loaded successfully.")
        # Optionally display loaded config details
        # st.sidebar.json(st.session_state.config)
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")
        st.stop() # Stop execution if config fails

# --- Chat Initialization Area ---
st.sidebar.header("Start New Chat")
initial_prompt_input = st.sidebar.text_area(
    "Enter the initial task for the agents:",
    height=150,
    key="initial_prompt_input",
    disabled=st.session_state.chat_initialized # Disable if chat already started
)

if st.sidebar.button("🚀 Start Chat", key="start_chat", disabled=st.session_state.chat_initialized or not initial_prompt_input):
    if not st.session_state.chat_initialized and initial_prompt_input:
        st.session_state.initial_prompt = initial_prompt_input # Store the prompt
        st.session_state.processing = True
        st.session_state.error_message = None
        try:
            with st.spinner("Initializing agents and starting chat..."):
                logger.info("Initializing chat...")
                (
                    st.session_state.manager,
                    st.session_state.boss_agent,
                    st.session_state.messages,
                    st.session_state.next_agent,
                ) = initialize_chat(st.session_state.config, st.session_state.initial_prompt)
                st.session_state.chat_initialized = True
                logger.info(f"Chat initialized. First message sent. Next agent: {st.session_state.next_agent.name if st.session_state.next_agent else 'None'}")
        except Exception as e:
            logger.error(f"Error initializing chat: {traceback.format_exc()}")
            st.session_state.error_message = f"Initialization failed: {e}"
        finally:
            st.session_state.processing = False
        st.rerun() # Rerun to update UI based on new state

# --- Display Error Message ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# --- Main Chat Interaction Area ---
chat_container = st.container() # Use a container for chat messages

with chat_container:
    # Display Chat History if Initialized
    if st.session_state.chat_initialized:
        display_messages(st.session_state.messages)

        # --- Interaction Controls ---
        # Check if there's a next agent and we are not currently processing a step
        if st.session_state.next_agent and not st.session_state.processing:
            next_agent_name = st.session_state.next_agent.name

            # Check if it's the User's (Boss's) turn
            if next_agent_name == BOSS_NAME:
                st.markdown(f"**Your turn (as {BOSS_NAME}):**")
                user_input = st.text_input(
                    "Enter your message:",
                    key=f"user_input_{len(st.session_state.messages)}", # Unique key needed
                    disabled=st.session_state.processing # Should always be False here, but keep for safety
                )
                if st.button("✉️ Send Message", key=f"send_{len(st.session_state.messages)}", disabled=st.session_state.processing or not user_input):
                    st.session_state.processing = True # Set processing BEFORE potentially long operation
                    st.session_state.error_message = None
                    should_rerun = False
                    # Use spinner for feedback while sending message
                    with st.spinner(f"Sending message as {BOSS_NAME}..."):
                        try:
                            logger.info(f"Sending user message: {user_input}")
                            new_messages, next_agent = send_user_message(
                                st.session_state.manager,
                                st.session_state.boss_agent,
                                user_input
                            )
                            st.session_state.messages.extend(new_messages)
                            st.session_state.next_agent = next_agent
                            logger.info(f"User message sent. Next agent: {next_agent.name if next_agent else 'None'}")
                            should_rerun = True # Rerun after successful send to process next turn
                        except Exception as e:
                             logger.error(f"Error sending user message: {traceback.format_exc()}")
                             st.session_state.error_message = f"Error sending message: {e}"
                        # Ensure processing is set to False even if error occurs
                        # No finally needed here as it's set *before* the rerun below

                    st.session_state.processing = False # Reset processing flag
                    if should_rerun:
                        st.rerun() # Rerun to display new messages and process the next agent's turn

            # --- Auto-run AI Agent's Turn ---
            else:
                # Automatically run the next AI agent's turn
                st.markdown(f"**Running turn for:** {next_agent_name}...") # Indicate which agent is running
                st.session_state.processing = True # Set processing flag BEFORE the operation
                st.session_state.error_message = None
                should_rerun = False

                # Use spinner for visual feedback during processing
                with st.spinner(f"Running {next_agent_name}'s turn..."):
                    try:
                        logger.info(f"Running step for agent: {next_agent_name}")
                        new_messages, next_agent = run_agent_step(
                            st.session_state.manager,
                            st.session_state.next_agent # Pass the actual agent object
                        )
                        st.session_state.messages.extend(new_messages)
                        st.session_state.next_agent = next_agent
                        logger.info(f"Agent {next_agent_name} finished. Next agent: {next_agent.name if next_agent else 'None'}")
                        should_rerun = True # Rerun after successful step
                    except Exception as e:
                        logger.error(f"Error during {next_agent_name}'s turn: {traceback.format_exc()}")
                        st.session_state.error_message = f"Error during {next_agent_name}'s turn: {e}"
                        # Stop the process on error by setting next_agent to None
                        st.session_state.next_agent = None
                        should_rerun = True # Rerun even on error to display the error message

                st.session_state.processing = False # Reset processing flag

                if should_rerun:
                     # Use st.rerun() to immediately trigger the next step check or display error
                     st.rerun()


        elif not st.session_state.next_agent and st.session_state.chat_initialized and not st.session_state.processing:
             # Displayed when the conversation is finished (next_agent is None)
             st.success("Chat finished.")
        # Note: If st.session_state.processing is True, this block is skipped, preventing
        # multiple triggers while an agent step or user send is in progress. The st.rerun()
        # call after processing ensures the UI updates and re-evaluates the state.

# Add a clear chat button or other controls as needed
if st.session_state.chat_initialized:
     if st.sidebar.button("Clear Chat History", key="clear_chat"):
         # Reset relevant state variables
         st.session_state.chat_initialized = False
         st.session_state.processing = False
         st.session_state.error_message = None
         # Keep config loaded
         st.session_state.manager = None
         st.session_state.boss_agent = None
         st.session_state.messages = []
         st.session_state.next_agent = None
         st.session_state.initial_prompt = ""
         # Clear UI elements associated with keys that might persist otherwise
         # (e.g., text inputs - though rerun should handle this)
         logger.info("Chat history cleared.")
         st.rerun()
