# Next Features Roadmap

This document outlines planned features for the Risk Management Challenge application, designed for incremental implementation.

## Feature 1: Separate Policy Input Field

**Goal:** Allow users to input the policy text separately from the task description.

**Implementation Steps:**

1.  **UI Modification (`app.py`):**
    *   Add a new `st.sidebar.text_area` widget labeled "Policy Text". Give it a unique key (e.g., `key="policy_text_input"`).
    *   Store the content of this text area in `st.session_state` using its key.
2.  **System Message Injection (`main.py`):**
    *   Modify the `setup_chat` function.
    *   Before creating the `PolicyGuard` agent:
        *   Retrieve the base system message content by reading `PolicyGuard.md`.
        *   Retrieve the `policy_text` from `st.session_state.policy_text_input`.
        *   If policy text exists in session state:
            *   Find the `## Policies` marker (or a suitable injection point) in the base system message string.
            *   Construct the new system message by injecting the `policy_text` under the marker.
        *   Else (no policy text provided), use the original base system message.
    *   Modify the call to `create_agent` for `PolicyGuard`. Instead of passing `system_message_file`, pass the constructed system message content directly using a new parameter in `create_agent` (see step 3).
3.  **Modify `create_agent` (`common_functions.py`):**
    *   Adjust the function signature to accept either a file path *or* direct content: `create_agent(name: str, llm_config: LLMConfiguration, system_message_file: Optional[str] = None, system_message_content: Optional[str] = None, agent_type="assistant")`.
    *   Inside `create_agent`:
        *   If `system_message_content` is provided, use it directly as `system_message`.
        *   Else if `system_message_file` is provided, call `read_system_message(system_message_file)` to get the `system_message`.
        *   Else, raise an error or use a default system message.
4.  **Update UI Labels:**
    *   Adjust the label for the original "initial task" input field to clarify it's for the "Task/Product Description" and should *not* include the policy.

---

## Feature 2: Persistent Task Description (Session-based)

**Goal:** Keep the task description entered by the user persistent within their current browser session, so it doesn't disappear on reruns before starting the chat.

**Implementation Steps:**

1.  **UI Modification (`app.py`):**
    *   Ensure the `st.sidebar.text_area` for the initial task description has a stable `key` (e.g., `key="initial_prompt_input"`). Streamlit automatically stores widget state using the key.
    *   *(No major changes needed if using keys properly, Streamlit handles session persistence for widget values)*.
    *   Optionally, to explicitly manage it: Read the value from `st.session_state.initial_prompt_input` when needed, instead of relying solely on the widget's immediate state upon button press.

---

## Feature 3: Editable Agent System Messages (Session-based)

**Goal:** Allow users to view and edit agent system messages within the UI for the current session, affecting agent behavior without modifying the underlying files.

**Implementation Steps:**

1.  **UI Modification (`app.py`):**
    *   Add a configuration section (e.g., `st.expander("Agent Configuration")` in the sidebar).
    *   For each agent (`PolicyGuard`, `Challenger`): # Updated name
        *   Define keys for session state, e.g., `policy_guard_sys_msg_key = "policy_guard_editable_prompt"`.
        *   On app startup (if not already in session state), read the *default* content from the agent's `.md` file (e.g., `PolicyGuard.md`) and store it in `st.session_state` under the defined key.
        *   Display a `st.text_area` inside the expander, labeled appropriately (e.g., "Edit PolicyGuard System Prompt").
        *   Set the `key` of the text area to the session state key (`policy_guard_sys_msg_key`). Streamlit will manage updating the session state from the text area.
2.  **Agent Creation Modification (`main.py`):**
    *   Modify the `setup_chat` function.
    *   Before creating each agent (e.g., `PolicyGuard`), retrieve the potentially edited system message from `st.session_state` using the key defined in the UI (e.g., `st.session_state.get(policy_guard_sys_msg_key)`).
    *   Call `create_agent`, passing the retrieved content via the `system_message_content` parameter (as defined in Feature 1, Step 3). Do *not* pass the `system_message_file` parameter if passing content directly.

*(Optional Extension: Add "Save to File" buttons next to each text area. The button's `on_click` handler would use `write_file` to save the content from the corresponding session state variable back to the agent's `.md` file.)*

---

## Feature 4: Input Length Monitoring & Warning

**Goal:** Provide feedback to the user about the token count of their inputs (policy, task) relative to the LLM's context window limit.

**Implementation Steps:**

1.  **Token Counting Setup:**
    *   Install a suitable token counting library (e.g., `pip install tiktoken` if using OpenAI models; for Gemini/Vertex AI, find the appropriate library or use an approximation like character count / 4).
    *   Import the library in `app.py`.
    *   Create a simple helper function `estimate_tokens(text: str) -> int` that uses the library or approximation.
2.  **Context Limit Definition:**
    *   Define a constant in `app.py` or load from config: `CONTEXT_LIMIT = <approx_token_limit_for_your_model>`. (e.g., 8192, 128000, etc.)
3.  **UI Integration (`app.py`):**
    *   Create a function `update_token_warning()`:
        *   Get the current text from the "Policy Text" and "Task/Product Description" text areas using their keys in `st.session_state`.
        *   Calculate the estimated token count for policy + task description using `estimate_tokens()`.
        *   (Optional but better) Estimate system prompt tokens (read from session state if editable, else estimate from default files).
        *   Sum these estimates.
        *   Display the current estimate: `st.sidebar.caption(f"Estimated Input Tokens: {total_tokens} / {CONTEXT_LIMIT}")`.
        *   If `total_tokens` exceeds a threshold (e.g., 85% of `CONTEXT_LIMIT`), display `st.sidebar.warning("Input length is approaching model context limit.")`. Otherwise, clear any existing warning (e.g., using an empty `st.empty()` placeholder).
    *   Add the `on_change=update_token_warning` argument to the `st.text_area` widgets for policy and task description.
    *   Call `update_token_warning()` once on script run to show initial state.

---
