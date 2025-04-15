import streamlit as st
import logging
import traceback

# Corrected imports
from main import BOSS_NAME, setup_chat # Import setup_chat from main, keep BOSS_NAME
from common_functions import (
    initiate_chat_task, # Import from common_functions
    run_agent_step,     # Import from common_functions
    send_user_message   # Import from common_functions
)

# Assuming load_config and config_path are defined elsewhere or will be fixed later
# Keep these imports for now, but they might cause errors if not properly defined/imported
# from main import load_config, config_path # Commented out for now, needs verification
# Placeholder - Define or import load_config and config_path appropriately
def load_config(path):
    # Example placeholder - replace with actual implementation
    print(f"Warning: Using placeholder load_config({path}). Replace with actual logic.")
    # In a real scenario, this would load config from a file (e.g., JSON, YAML)
    # For now, let's assume it returns a dictionary needed by setup_chat
    # This structure needs to align with what setup_chat expects
    return {
        "llm_provider": "VERTEX_AI", # Defaulting to Vertex AI as per main.py logic
        "model_name": "gemini-1.5-pro-002" # Defaulting model
    }
config_path = "config.json" # Example placeholder path


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Constants and Configuration ---
MAX_MESSAGES_DISPLAY = 50 # Limit messages displayed to prevent clutter
POLICY_TEXT_KEY = "policy_text_input" # Key for the policy text area
TASK_PROMPT_KEY = "initial_prompt_input" # Key for the task description area

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
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        elif not isinstance(content, str):
             content = str(content) # Ensure content is a string

        # Simple way to distinguish user (Boss) messages - adjust if needed
        if sender_name == BOSS_NAME:
            with st.chat_message("user", avatar="🧑"): # Or use a specific Boss avatar
                 st.markdown(f"""**{sender_name}:**\n{content}""")
        else:
            is_agent_message = "sender" in msg or ("role" in msg and msg["role"] not in ["system", "tool", "function"])
            if is_agent_message:
                 with st.chat_message("assistant", avatar="🤖"): # Generic AI avatar
                      st.markdown(f"""**{sender_name}:**\n{content}""")
            else: # Likely a system message or tool call/result
                 with st.chat_message("system", avatar="⚙️"):
                      st.markdown(f"""_{sender_name}: {content}_""") # Italicize system/tool messages


# --- Streamlit App UI ---
st.title("🤖 Risk Management Challenge Session with human Product Lead and two AI agents")

# --- Initialization ---
# Initialize session state variables if they don't exist
default_values = {
    "chat_initialized": False,
    "processing": False,
    "error_message": None,
    "config": None,
    "manager": None,
    "boss_agent": None,
    "messages": [],
    "next_agent": None,
    TASK_PROMPT_KEY: "", # Initialize task prompt
    POLICY_TEXT_KEY: "", # Initialize policy text
    # Add other keys if needed
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Configuration Loading (using placeholder) ---
if not st.session_state.config:
    try:
        # Using placeholder load_config and config_path defined above
        st.session_state.config = load_config(config_path)
        st.sidebar.success("Configuration loaded (using placeholder). Update required.")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration (placeholder): {e}")
        st.stop()

# --- Chat Setup Area (Moved out of button click to happen once) ---
# Setup agents and manager using the loaded config
# Note: This setup happens on every rerun BEFORE the start button is processed,
# which means PolicyGuard's system message will be constructed based on the
# *current* value in the policy text area *before* the chat officially starts.
if not st.session_state.manager and st.session_state.config:
    try:
        with st.spinner("Setting up agents based on configuration..."):
             logger.info("Setting up chat components...")
             # Pass policy text from session state to setup_chat
             st.session_state.manager, st.session_state.boss_agent = setup_chat(
                 llm_provider=st.session_state.config.get("llm_provider", "VERTEX_AI"),
                 model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                 # Pass policy text from session state (might be empty initially)
                 policy_text=st.session_state.get(POLICY_TEXT_KEY, "")
             )
             logger.info("Chat components (manager, boss_agent) set up.")
    except Exception as e:
         logger.error(f"Error setting up chat components: {traceback.format_exc()}")
         st.session_state.error_message = f"Setup failed: {e}"
         # Clear potentially partially initialized state
         st.session_state.manager = None
         st.session_state.boss_agent = None
         st.session_state.chat_initialized = False # Prevent starting chat

# --- Start Chat Area (Button triggers initiation) ---
st.sidebar.header("Start New Chat")

# --- NEW: Policy Text Input ---
policy_text_input = st.sidebar.text_area(
    "Enter the Policy Text:",
    height=100,
    key=POLICY_TEXT_KEY, # Use defined key
    # Disable if chat is running OR if setup failed (no manager)
    disabled=st.session_state.chat_initialized or not st.session_state.manager,
    help="Enter the policy content here. This will be injected into the PolicyGuard agent's instructions."
)

# --- UPDATED: Task Description Input ---
initial_prompt_input = st.sidebar.text_area(
    "Enter the Task/Product Description:", # Updated label
    height=150,
    key=TASK_PROMPT_KEY, # Use defined key
    # Disable if chat is running OR if setup failed (no manager)
    disabled=st.session_state.chat_initialized or not st.session_state.manager,
    help="Describe the product or task. Do NOT include the policy text here." # Added help text
)

if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    # Disable if chat running, no TASK prompt, or setup failed
                    disabled=st.session_state.chat_initialized
                             or not st.session_state.get(TASK_PROMPT_KEY) # Check task prompt specifically
                             or not st.session_state.manager):

    # Check conditions again inside the button logic for safety
    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    if not st.session_state.chat_initialized and task_prompt and st.session_state.manager and st.session_state.boss_agent:
        # Policy text is already in session state due to the widget
        st.session_state.processing = True
        st.session_state.error_message = None
        try:
            with st.spinner("Initiating chat task..."):
                logger.info("Initiating chat task...")
                # Pass only the TASK description to initiate_chat_task
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.boss_agent,
                    st.session_state.manager,
                    task_prompt # Pass only the task description from session state
                )
                # Update session state
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True
                logger.info(f"Chat initiated. Task prompt sent. Next agent: {st.session_state.next_agent.name if st.session_state.next_agent else 'None'}")

        except Exception as e:
            logger.error(f"Error initiating chat task: {traceback.format_exc()}")
            st.session_state.error_message = f"Initiation failed: {e}"
            st.session_state.chat_initialized = False # Ensure chat doesn't appear initialized
        finally:
            st.session_state.processing = False
        st.rerun() # Rerun to display the first message and potentially the next agent's turn

# --- Display Error Message ---
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# --- Main Chat Interaction Area ---
chat_container = st.container()

with chat_container:
    if st.session_state.chat_initialized:
        display_messages(st.session_state.messages)

        if st.session_state.next_agent and not st.session_state.processing:
            next_agent_name = st.session_state.next_agent.name

            if next_agent_name == BOSS_NAME:
                st.markdown(f"**Your turn (as {BOSS_NAME}):**")
                # Use st.form for Boss input
                with st.form(key=f'boss_input_form_{len(st.session_state.messages)}'): # Unique key per turn
                    user_input = st.text_input(
                        "Enter your message:",
                        key=f"user_input_{len(st.session_state.messages)}", # Keep unique key for input
                        disabled=st.session_state.processing,
                        placeholder="Type your message and press Enter to send..." # Added placeholder
                    )
                    # Replace st.button with st.form_submit_button
                    submitted = st.form_submit_button(
                        "✉️ Send Message",
                        disabled=st.session_state.processing
                    )
                    # Logic executed upon form submission (Enter key or button click)
                    if submitted:
                        if not user_input: # Prevent sending empty messages
                             st.warning("Please enter a message.")
                             st.stop() # Stop execution for this run to prevent processing empty input
                        else:
                            st.session_state.processing = True
                            st.session_state.error_message = None
                            should_rerun = False
                            with st.spinner(f"Sending message as {BOSS_NAME}..."):
                                try:
                                    logger.info(f"Sending user message: {user_input}")
                                    # Use send_user_message from common_functions
                                    new_messages, next_agent = send_user_message(
                                        st.session_state.manager,
                                        st.session_state.boss_agent,
                                        user_input # Value is accessible after form submission
                                    )
                                    st.session_state.messages.extend(new_messages)
                                    st.session_state.next_agent = next_agent
                                    logger.info(f"User message sent. Next agent: {next_agent.name if next_agent else 'None'}")
                                    should_rerun = True
                                except Exception as e:
                                     logger.error(f"Error sending user message: {traceback.format_exc()}")
                                     st.session_state.error_message = f"Error sending message: {e}"

                            st.session_state.processing = False
                            if should_rerun:
                                st.rerun() # Rerun to display new messages and next step

            else: # Auto-run AI Agent's Turn
                st.markdown(f"**Running turn for:** {next_agent_name}...")
                st.session_state.processing = True
                st.session_state.error_message = None
                should_rerun = False
                with st.spinner(f"Running {next_agent_name}'s turn..."):
                    try:
                        logger.info(f"Running step for agent: {next_agent_name}")
                        # Use run_agent_step from common_functions
                        new_messages, next_agent = run_agent_step(
                            st.session_state.manager,
                            st.session_state.next_agent
                        )
                        st.session_state.messages.extend(new_messages)
                        st.session_state.next_agent = next_agent
                        logger.info(f"Agent {next_agent_name} finished. Next agent: {next_agent.name if next_agent else 'None'}")
                        should_rerun = True
                    except Exception as e:
                        logger.error(f"Error during {next_agent_name}'s turn: {traceback.format_exc()}")
                        st.session_state.error_message = f"Error during {next_agent_name}'s turn: {e}"
                        st.session_state.next_agent = None # Stop on error
                        should_rerun = True

                st.session_state.processing = False
                if should_rerun:
                     st.rerun()

        elif not st.session_state.next_agent and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished.")

# --- Clear Chat Button ---
if st.session_state.chat_initialized or st.session_state.error_message: # Show clear button if chat started or error occurred
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         # Reset state, keeping config and agent setup if possible
         st.session_state.chat_initialized = False
         st.session_state.processing = False
         st.session_state.error_message = None
         st.session_state.messages = []
         st.session_state.next_agent = None
         st.session_state[TASK_PROMPT_KEY] = "" # Clear task prompt
         st.session_state[POLICY_TEXT_KEY] = "" # Clear policy text

         # Re-run setup to ensure agents are ready for the next chat
         # This relies on setup_chat being safe to call multiple times if manager already exists
         # If setup_chat is expensive or stateful in a problematic way, this might need adjustment
         # For now, assume it's okay to re-initialize the manager/boss agent pointers
         st.session_state.manager = None
         st.session_state.boss_agent = None

         logger.info("Chat state cleared. Re-running setup.")
         st.rerun()
