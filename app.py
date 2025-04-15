import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional

# Import AutoGen components directly in app.py
import autogen
from autogen import UserProxyAgent, GroupChatManager, GroupChat

# Import necessary components from local modules
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC
from common_functions import (
    create_agent,         # Still needed
    create_groupchat,     # Still needed
    create_groupchat_manager, # Still needed
    initiate_chat_task, # Import from common_functions
    run_agent_step,     # Import from common_functions
    send_user_message   # Import from common_functions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants (Moved from main.py) ---
BOSS_NAME = "Boss"
POLICY_GUARD_NAME = "PolicyGuard"
CHALLENGER_NAME = "FirstLineChallenger"

BOSS_SYS_MSG_FILE = "Boss.md"
POLICY_GUARD_SYS_MSG_FILE = "PolicyGuard.md"
CHALLENGER_SYS_MSG_FILE = "FirstLineChallenger.md"

# Marker for policy injection
POLICY_INJECTION_MARKER = "## Policies"

MAX_MESSAGES_DISPLAY = 50 # Limit messages displayed to prevent clutter
POLICY_TEXT_KEY = "policy_text_input" # Key for the policy text area
TASK_PROMPT_KEY = "initial_prompt_input" # Key for the task description area

# --- Helper Functions (Moved/Adapted from main.py and app.py) ---

def _read_system_message(file_path: str) -> str:
    """Reads system message from a file, trying relative path first, then script directory."""
    try:
        # Try relative path first
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        try:
            # Try path relative to the script directory as a fallback
            script_dir = os.path.dirname(__file__)
            alt_path = os.path.join(script_dir, file_path)
            with open(alt_path, "r", encoding="utf-8") as f:
                logger.warning(f"Could not find {file_path} directly, reading from {alt_path}")
                return f.read()
        except FileNotFoundError:
             logger.error(f"System message file not found at {file_path} or {alt_path}")
             raise FileNotFoundError(f"Agent system message file not found: {file_path}")
        except Exception as e:
             logger.error(f"Error reading system message file {file_path} (or alt): {e}", exc_info=True)
             raise # Re-raise other errors

def setup_chat(
    llm_provider: str = VERTEX_AI,
    model_name: str = "gemini-1.5-pro-002",
    policy_text: Optional[str] = None
    ) -> Tuple[GroupChatManager, UserProxyAgent]:
    """
    Sets up the agents, group chat, and manager based on selected LLM provider.
    Injects provided policy_text into PolicyGuard's system message.
    Reads Vertex AI credentials from st.secrets.
    (Function moved from main.py to app.py)
    """
    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")
    if policy_text:
        logger.info("Policy text provided, will inject into PolicyGuard.")
    else:
        logger.info("No policy text provided, PolicyGuard will use default from file.")

    # --- LLM Configuration ---
    if llm_provider == VERTEX_AI:
        try:
            if "gcp_credentials" not in st.secrets:
                 raise ValueError("Missing 'gcp_credentials' section in Streamlit secrets.")
            required_keys = ["project_id", "private_key", "client_email", "type"]
            if not all(key in st.secrets["gcp_credentials"] for key in required_keys):
                 raise ValueError(f"Missing required keys ({required_keys}) within 'gcp_credentials' in Streamlit secrets.")
            vertex_credentials = dict(st.secrets["gcp_credentials"])
            llm_config = LLMConfiguration(
                VERTEX_AI,
                model_name,
                project_id=vertex_credentials.get('project_id'),
                location="us-central1",
                vertex_credentials=vertex_credentials,
            )
            logger.info("Vertex AI LLM configuration loaded from st.secrets.")
        except ValueError as e:
             logger.error(f"Credential error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error loading Vertex AI credentials from st.secrets: {e}", exc_info=True)
            raise
    # Add elif blocks here for AZURE, ANTHROPIC etc. if needed
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    if not llm_config.get_config():
        raise ValueError("Failed to create a valid LLM configuration object or dictionary.")

    # --- Agent Creation ---
    try:
        for file_path in [BOSS_SYS_MSG_FILE, CHALLENGER_SYS_MSG_FILE, POLICY_GUARD_SYS_MSG_FILE]:
             _read_system_message(file_path) # Check if files are readable early

        boss = create_agent(
            name=BOSS_NAME,
            llm_config=llm_config,
            system_message_file=BOSS_SYS_MSG_FILE,
            agent_type="user_proxy",
        )

        base_policy_guard_sys_msg = _read_system_message(POLICY_GUARD_SYS_MSG_FILE)
        policy_guard_final_sys_msg = base_policy_guard_sys_msg
        if policy_text and policy_text.strip():
            if POLICY_INJECTION_MARKER in base_policy_guard_sys_msg:
                policy_guard_final_sys_msg = base_policy_guard_sys_msg.replace(
                    POLICY_INJECTION_MARKER,
                    f"""{POLICY_INJECTION_MARKER}

{policy_text.strip()}""",
                    1
                )
                logger.info(f"Injected policy text into PolicyGuard system message under '{POLICY_INJECTION_MARKER}'.")
            else:
                logger.warning(f"Policy injection marker '{POLICY_INJECTION_MARKER}' not found in {POLICY_GUARD_SYS_MSG_FILE}. Appending policy text instead.")
                policy_guard_final_sys_msg += f"""

## Policies

{policy_text.strip()}"""
        else:
             logger.info(f"Using default system message for PolicyGuard from {POLICY_GUARD_SYS_MSG_FILE}.")

        policy_guard = create_agent(
            name=POLICY_GUARD_NAME,
            llm_config=llm_config,
            system_message_content=policy_guard_final_sys_msg,
            system_message_file=None
        )

        first_line_challenger = create_agent(
            name=CHALLENGER_NAME,
            llm_config=llm_config,
            system_message_file=CHALLENGER_SYS_MSG_FILE,
        )

        logger.info("Agents created successfully.")
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except ValueError as e:
        logger.error(f"Agent creation failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during agent creation: {e}", exc_info=True)
        raise

    # --- Group Chat Setup ---
    policy_team = [boss, policy_guard, first_line_challenger]
    try:
        groupchat = create_groupchat(policy_team, max_round=50)
        logger.info("GroupChat created successfully.")
    except ValueError as e:
        logger.error(f"GroupChat creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during GroupChat creation: {e}", exc_info=True)
        raise

    manager_llm_config = llm_config # Reusing the same config
    try:
        manager = create_groupchat_manager(groupchat, manager_llm_config)
        logger.info("GroupChatManager created successfully.")
    except ValueError as e:
        logger.error(f"GroupChatManager creation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during GroupChatManager creation: {e}", exc_info=True)
        raise

    logger.info("Chat setup completed.")
    return manager, boss

def display_messages(messages):
    """Displays chat messages, limiting the number shown."""
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY:
        st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    # Display relevant messages
    for i, msg in enumerate(messages[start_index:], start=start_index):
        sender_name = msg.get("name", "System")
        if not sender_name and "role" in msg:
             sender_name = msg["role"].capitalize()

        content = msg.get("content", "")
        if isinstance(content, list):
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
             content = str(content)

        if sender_name == BOSS_NAME:
            with st.chat_message("user", avatar="🧑"):
                 st.markdown(f"""**{sender_name}:**\n{content}""")
        else:
            is_agent_message = "sender" in msg or ("role" in msg and msg["role"] not in ["system", "tool", "function"])
            if is_agent_message:
                 with st.chat_message("assistant", avatar="🤖"):
                      st.markdown(f"""**{sender_name}:**\n{content}""")
            else:
                 with st.chat_message("system", avatar="⚙️"):
                      st.markdown(f"""_{sender_name}: {content}_""")

# Placeholder for load_config - Replace with actual implementation if needed
def load_config(path):
    print(f"Warning: Using placeholder load_config({path}). Replace with actual logic.")
    # Example: Default to Vertex AI if not loading from a real config file
    return {
        "llm_provider": "VERTEX_AI",
        "model_name": "gemini-1.5-pro-002"
    }
config_path = "config.json" # Example placeholder path

# --- Streamlit App UI ---
st.title("🤖 Risk Management Challenge Session")

# --- Initialization ---
default_values = {
    "chat_initialized": False,
    "processing": False,
    "error_message": None,
    "config": None,
    "manager": None,
    "boss_agent": None,
    "messages": [],
    "next_agent": None,
    TASK_PROMPT_KEY: "",
    POLICY_TEXT_KEY: "",
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Configuration Loading ---
if not st.session_state.config:
    try:
        st.session_state.config = load_config(config_path)
        st.sidebar.success("Configuration loaded (using placeholder). Update required.")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration (placeholder): {e}")
        st.stop()

# --- Chat Setup Area ---
# This now uses the setup_chat function defined within app.py
if not st.session_state.manager and st.session_state.config:
    try:
        with st.spinner("Setting up agents based on configuration..."):
             logger.info("Setting up chat components...")
             st.session_state.manager, st.session_state.boss_agent = setup_chat(
                 llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                 model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                 policy_text=st.session_state.get(POLICY_TEXT_KEY, "")
             )
             logger.info("Chat components (manager, boss_agent) set up.")
    except Exception as e:
         logger.error(f"Error setting up chat components: {traceback.format_exc()}")
         st.session_state.error_message = f"Setup failed: {e}"
         st.session_state.manager = None
         st.session_state.boss_agent = None
         st.session_state.chat_initialized = False

# --- Start Chat Area ---
st.sidebar.header("Start New Chat")

policy_text_input = st.sidebar.text_area(
    "Enter the Policy Text:",
    height=100,
    key=POLICY_TEXT_KEY,
    disabled=st.session_state.chat_initialized or not st.session_state.manager,
    help="Enter the policy content here. This will be injected into the PolicyGuard agent's instructions."
)

initial_prompt_input = st.sidebar.text_area(
    "Enter the Task/Product Description:",
    height=150,
    key=TASK_PROMPT_KEY,
    disabled=st.session_state.chat_initialized or not st.session_state.manager,
    help="Describe the product or task. Do NOT include the policy text here."
)

if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    disabled=st.session_state.chat_initialized
                             or not st.session_state.get(TASK_PROMPT_KEY)
                             or not st.session_state.manager):

    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    if not st.session_state.chat_initialized and task_prompt and st.session_state.manager and st.session_state.boss_agent:
        st.session_state.processing = True
        st.session_state.error_message = None
        try:
            with st.spinner("Initiating chat task..."):
                logger.info("Initiating chat task...")
                # Need to re-run setup here IF policy can change after initial load
                # and before starting chat. If setup is fast, it's simpler.
                # Alternatively, modify PolicyGuard agent directly if possible.
                # For now, assume re-running setup is acceptable.
                logger.info("Re-running setup with potentially updated policy before starting chat...")
                st.session_state.manager, st.session_state.boss_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    policy_text=st.session_state.get(POLICY_TEXT_KEY, "") # Use current policy text
                )
                logger.info("Setup complete. Initiating chat task...")

                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.boss_agent,
                    st.session_state.manager,
                    task_prompt
                )
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent
                st.session_state.chat_initialized = True
                logger.info(f"Chat initiated. Task prompt sent. Next agent: {st.session_state.next_agent.name if st.session_state.next_agent else 'None'}")

        except Exception as e:
            logger.error(f"Error initiating chat task: {traceback.format_exc()}")
            st.session_state.error_message = f"Initiation failed: {e}"
            st.session_state.chat_initialized = False
        finally:
            st.session_state.processing = False
        st.rerun()

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
                with st.form(key=f'boss_input_form_{len(st.session_state.messages)}'):
                    user_input = st.text_input(
                        "Enter your message:",
                        key=f"user_input_{len(st.session_state.messages)}",
                        disabled=st.session_state.processing,
                        placeholder="Type your message and press Enter to send..."
                    )
                    submitted = st.form_submit_button(
                        "✉️ Send Message",
                        disabled=st.session_state.processing
                    )
                    if submitted:
                        if not user_input:
                             st.warning("Please enter a message.")
                             st.stop()
                        else:
                            st.session_state.processing = True
                            st.session_state.error_message = None
                            should_rerun = False
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
                                    should_rerun = True
                                except Exception as e:
                                     logger.error(f"Error sending user message: {traceback.format_exc()}")
                                     st.session_state.error_message = f"Error sending message: {e}"

                            st.session_state.processing = False
                            if should_rerun:
                                st.rerun()

            else: # Auto-run AI Agent's Turn
                st.markdown(f"**Running turn for:** {next_agent_name}...")
                st.session_state.processing = True
                st.session_state.error_message = None
                should_rerun = False
                with st.spinner(f"Running {next_agent_name}'s turn..."):
                    try:
                        logger.info(f"Running step for agent: {next_agent_name}")
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
                        st.session_state.next_agent = None
                        should_rerun = True

                st.session_state.processing = False
                if should_rerun:
                     st.rerun()

        elif not st.session_state.next_agent and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished.")

# --- Clear Chat Button ---
if st.session_state.chat_initialized or st.session_state.error_message:
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         st.session_state.chat_initialized = False
         st.session_state.processing = False
         st.session_state.error_message = None
         st.session_state.messages = []
         st.session_state.next_agent = None
         st.session_state[TASK_PROMPT_KEY] = ""
         st.session_state[POLICY_TEXT_KEY] = ""

         st.session_state.manager = None
         st.session_state.boss_agent = None

         logger.info("Chat state cleared. Re-running setup on next interaction.")
         st.rerun()
