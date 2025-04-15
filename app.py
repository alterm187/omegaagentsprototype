import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional, Dict

# Import AutoGen components directly in app.py
import autogen
from autogen import UserProxyAgent, GroupChatManager, GroupChat, Agent

# Import necessary components from local modules
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC
from common_functions import (
    create_agent,         # Still needed
    create_groupchat,     # Still needed
    create_groupchat_manager, # Still needed
    initiate_chat_task, # Import from common_functions
    run_agent_step,     # Import from common_functions
    send_user_message,   # Import from common_functions
    read_system_message # Now needed in app.py
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---

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

# Keys for editable system messages
POLICY_GUARD_EDIT_KEY = "policy_guard_editable_prompt"
CHALLENGER_EDIT_KEY = "challenger_editable_prompt"
BOSS_EDIT_KEY = "boss_editable_prompt"

AGENT_CONFIG = {
    POLICY_GUARD_NAME: {"file": POLICY_GUARD_SYS_MSG_FILE, "key": POLICY_GUARD_EDIT_KEY},
    CHALLENGER_NAME: {"file": CHALLENGER_SYS_MSG_FILE, "key": CHALLENGER_EDIT_KEY},
    BOSS_NAME: {"file": BOSS_SYS_MSG_FILE, "key": BOSS_EDIT_KEY},
}

# Feature 4: Context Limit for Gemini Pro 1.5 (approximate)
CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85 # Warn at 85% of context limit

# --- Helper Functions ---

def estimate_tokens(text: str) -> int:
    """Approximates token count using character count / 4."""
    return len(text or "") // 4

def _read_system_message(file_path: str) -> str:
    """Reads system message from a file, using the function from common_functions."""
    try:
        # Try relative path first (common_functions handles this)
        return read_system_message(file_path)
    except FileNotFoundError:
         logger.error(f"System message file not found: {file_path}")
         raise
    except Exception as e:
         logger.error(f"Error reading system message file {file_path}: {e}", exc_info=True)
         raise # Re-raise other errors

def initialize_editable_prompts():
    """Loads default agent prompts into session state if they don't exist."""
    for agent_name, config in AGENT_CONFIG.items():
        if config["key"] not in st.session_state:
            try:
                st.session_state[config["key"]] = _read_system_message(config["file"])
                logger.info(f"Loaded default system message for {agent_name} into session state ({config['key']}).")
            except Exception as e:
                st.session_state[config["key"]] = f"Error loading default from {config['file']}: {e}"
                logger.error(f"Failed to load initial prompt for {agent_name} from {config['file']}: {e}")

def update_token_warning():
    """Calculates estimated total tokens and displays warning if near limit."""
    policy_text = st.session_state.get(POLICY_TEXT_KEY, "")
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    policy_guard_prompt = st.session_state.get(POLICY_GUARD_EDIT_KEY, "")
    challenger_prompt = st.session_state.get(CHALLENGER_EDIT_KEY, "")
    boss_prompt = st.session_state.get(BOSS_EDIT_KEY, "")

    # Estimate tokens for each part
    policy_tokens = estimate_tokens(policy_text)
    task_tokens = estimate_tokens(task_text)
    policy_guard_tokens = estimate_tokens(policy_guard_prompt)
    challenger_tokens = estimate_tokens(challenger_prompt)
    boss_tokens = estimate_tokens(boss_prompt)

    total_system_prompt_tokens = policy_guard_tokens + challenger_tokens + boss_tokens
    total_input_tokens = policy_tokens + task_tokens
    total_estimated_tokens = total_input_tokens + total_system_prompt_tokens

    # Update caption (using global placeholder defined later)
    if 'token_info_placeholder' in globals():
        token_info_placeholder.caption(f"Estimated Input Tokens: ~{total_estimated_tokens:,} / {CONTEXT_LIMIT:,}")

    # Update warning (using global placeholder defined later)
    if 'token_warning_placeholder' in globals():
        if total_estimated_tokens > CONTEXT_LIMIT * WARNING_THRESHOLD:
            token_warning_placeholder.warning(f"Inputs approaching context limit ({WARNING_THRESHOLD*100:.0f}%). Total: ~{total_estimated_tokens:,}")
        else:
            token_warning_placeholder.empty() # Clear warning

def setup_chat(
    llm_provider: str = VERTEX_AI,
    model_name: str = "gemini-1.5-pro-002",
    policy_text: Optional[str] = None,
    agent_prompts: Dict[str, str] = {} # Dict mapping agent name to system prompt content
    ) -> Tuple[GroupChatManager, UserProxyAgent]:
    """
    Sets up the agents, group chat, and manager based on selected LLM provider.
    Uses provided agent prompts from session state.
    Injects provided policy_text into PolicyGuard's system message.
    Reads Vertex AI credentials from st.secrets.
    """
    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")
    if policy_text:
        logger.info("Policy text provided, will inject into PolicyGuard.")
    else:
        logger.info("No policy text provided, PolicyGuard will use prompt from session state.")

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

    # --- Agent Creation (Using Session State Prompts) ---

    try:
        agents = {}
        for agent_name, config in AGENT_CONFIG.items():
            system_message_content = agent_prompts.get(agent_name)
            if not system_message_content:
                 logger.error(f"Missing system message content for agent {agent_name} in setup_chat call.")
                 # Fallback to reading file again as a safety measure, though it shouldn't happen if initialized correctly
                 try:
                     system_message_content = _read_system_message(config["file"])
                     logger.warning(f"Had to re-read {config['file']} for {agent_name} during setup.")
                 except Exception as e:
                     raise ValueError(f"Could not load system message for {agent_name}: {e}")

            # Inject policy into PolicyGuard's message
            if agent_name == POLICY_GUARD_NAME and policy_text and policy_text.strip():
                 base_policy_guard_sys_msg = system_message_content # Start with the editable content
                 if POLICY_INJECTION_MARKER in base_policy_guard_sys_msg:
                     system_message_content = base_policy_guard_sys_msg.replace(
                         POLICY_INJECTION_MARKER,
                         f"""{POLICY_INJECTION_MARKER}

{policy_text.strip()}""",
                         1
                     )
                     logger.info(f"Injected policy text into editable PolicyGuard system message under '{POLICY_INJECTION_MARKER}'.")
                 else:
                     logger.warning(f"Policy injection marker '{POLICY_INJECTION_MARKER}' not found in editable PolicyGuard prompt. Appending policy text instead.")
                     system_message_content += f"""

## Policies

{policy_text.strip()}"""

            agent_type = "user_proxy" if agent_name == BOSS_NAME else "assistant"
            agents[agent_name] = create_agent(
                name=agent_name,
                llm_config=llm_config,
                system_message_content=system_message_content, # Pass the content directly
                system_message_file=None, # Ensure file path is not used
                agent_type=agent_type,
            )

        boss = agents[BOSS_NAME]
        policy_guard = agents[POLICY_GUARD_NAME]
        first_line_challenger = agents[CHALLENGER_NAME]

        logger.info("Agents created successfully using session state prompts.")
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
            with st.chat_message("Boss", avatar="🧑"):
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

st.title("🤖 Risk Management Challenge Session with ProductLead and two AI agents")

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
    # Editable prompts will be initialized by function below
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Initialize Editable Prompts (Run Once) ---

initialize_editable_prompts()

# --- Configuration Loading ---

if not st.session_state.config:
    try:
        st.session_state.config = load_config(config_path)
        st.sidebar.success("Configuration loaded (using placeholder). Update required.")
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration (placeholder): {e}")
        st.stop()

# --- Chat Setup Area (Only if Manager doesn't exist) ---

# We now delay the setup until "Start Chat" is pressed to ensure edited prompts are used.
if not st.session_state.manager and st.session_state.config:
     # Show a placeholder or indicator that setup will happen on start
     st.sidebar.info("Agents will be configured when chat starts.")
     pass # Don't set up manager/boss here yet

# --- Agent Configuration Expander (Sidebar) ---

with st.sidebar.expander("Agent Configuration"):
    st.caption("Edit agent system prompts for this session:")
    for agent_name, config_info in AGENT_CONFIG.items():
        st.text_area(
            f"Edit {agent_name} System Prompt",
            key=config_info["key"], # Link to session state key
            height=150,
            disabled=st.session_state.chat_initialized, # Disable after chat starts
            help=f"Modify the instructions for the {agent_name} agent.",
            on_change=update_token_warning # Add on_change hook
        )
        # Optional: Add 'Save to File' button here if desired later

# --- Start Chat Area ---

st.sidebar.header("Start New Chat")

# Create placeholders for token info/warnings *before* the text areas
token_info_placeholder = st.sidebar.empty()
token_warning_placeholder = st.sidebar.empty()

policy_text_input = st.sidebar.text_area(
    "Enter the Policy Text:",
    height=100,
    key=POLICY_TEXT_KEY,
    disabled=st.session_state.chat_initialized or not st.session_state.config, # Disable if no config or chat started
    help="Enter the policy content here. This will be injected into the PolicyGuard agent's instructions.",
    on_change=update_token_warning # Add on_change hook
)

initial_prompt_input = st.sidebar.text_area(
    "Enter the Task/Product Description:",
    height=150,
    key=TASK_PROMPT_KEY,
    disabled=st.session_state.chat_initialized or not st.session_state.config, # Disable if no config or chat started
    help="Describe the product or task. Do NOT include the policy text here.",
    on_change=update_token_warning # Add on_change hook
)

# Initial call to display token info/warning based on default/current values
update_token_warning()

if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    disabled=st.session_state.chat_initialized
                             or not st.session_state.get(TASK_PROMPT_KEY)
                             or not st.session_state.config): # Check config instead of manager

    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    if not st.session_state.chat_initialized and task_prompt and st.session_state.config:
        st.session_state.processing = True
        st.session_state.error_message = None
        try:
            with st.spinner("Setting up agents and initiating chat task..."):
                logger.info("Setting up chat components with potentially updated prompts and policy...")

                # Get current prompts from session state
                current_agent_prompts = {}
                for agent_name, config_info in AGENT_CONFIG.items():
                    current_agent_prompts[agent_name] = st.session_state.get(config_info["key"], f"Error: Missing prompt for {agent_name}")

                # Setup chat using current prompts and policy
                st.session_state.manager, st.session_state.boss_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    policy_text=st.session_state.get(POLICY_TEXT_KEY, ""), # Use current policy text
                    agent_prompts=current_agent_prompts # Pass the editable prompts
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
            logger.error(f"Error setting up or initiating chat task: {traceback.format_exc()}")
            st.session_state.error_message = f"Setup/Initiation failed: {e}"
            st.session_state.chat_initialized = False
            # Reset manager/boss if setup failed
            st.session_state.manager = None
            st.session_state.boss_agent = None
        finally:
            st.session_state.processing = False
        st.rerun()

# --- Display Error Message ---

if st.session_state.error_message:
    st.error(st.session_state.error_message)

# --- Main Chat Interaction Area ---

chat_container = st.container()

with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager: # Check manager exists
        display_messages(st.session_state.messages)

        if st.session_state.next_agent and not st.session_state.processing:
            next_agent_name = st.session_state.next_agent.name

            if next_agent_name == BOSS_NAME:
                st.markdown(f"**Your turn (as {BOSS_NAME}):**")
                # Use a unique key based on message length to avoid state issues on rerun
                form_key = f'boss_input_form_{len(st.session_state.messages)}'
                input_key = f"user_input_{len(st.session_state.messages)}"

                with st.form(key=form_key):
                    user_input = st.text_input(
                        "Enter your message:",
                        key=input_key,
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
                             # No rerun needed, just stay waiting for input
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
                                     should_rerun = True # Rerun to show error

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
                        st.session_state.next_agent = None # Stop the chat on agent error
                        should_rerun = True

                st.session_state.processing = False
                if should_rerun:
                     st.rerun()

        elif not st.session_state.next_agent and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished.")

# --- Clear Chat Button ---

if st.session_state.chat_initialized or st.session_state.error_message or not st.session_state.manager: # Allow reset if setup failed
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         # Reset core state
         st.session_state.chat_initialized = False
         st.session_state.processing = False
         st.session_state.error_message = None
         st.session_state.messages = []
         st.session_state.next_agent = None
         st.session_state.manager = None
         st.session_state.boss_agent = None

         # Clear inputs *but keep editable prompts*
         st.session_state[TASK_PROMPT_KEY] = ""
         st.session_state[POLICY_TEXT_KEY] = ""

         # # Optional: Reset editable prompts to default (Uncomment if desired)
         # for agent_name, config in AGENT_CONFIG.items():
         #     try:
         #         st.session_state[config["key"]] = _read_system_message(config["file"])
         #     except Exception as e:
         #         st.session_state[config["key"]] = f"Error reloading default from {config['file']}: {e}"
         #         logger.error(f"Failed to reload prompt for {agent_name} during reset: {e}")

         logger.info("Chat state cleared. Ready for new configuration/start.")
         # Also manually trigger token update on reset
         update_token_warning()
         st.rerun()

