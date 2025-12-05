import os
import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import json
import pandas as pd


# Cache expensive operations to improve performance
@st.cache_resource
def get_cached_embedding_model():
    """Cache embedding model to avoid reloading on every Streamlit rerun."""
    from rag_backend import get_embedding_model
    return get_embedding_model()


@st.cache_resource
def get_cached_vector_store():
    """Cache vector store to avoid reloading on every Streamlit rerun."""
    from rag_backend import get_vector_store
    return get_vector_store()


@st.cache_resource
def get_cached_agent():
    """Cache agent creation to avoid reloading LLM on every query."""
    from agent import create_document_agent
    return create_document_agent()


def reset_streamlit_caches():
    """Reset Streamlit caches after DB updates."""
    st.cache_resource.clear()
    from rag_backend import reset_vector_store
    reset_vector_store()



class StreamlitWebInterface:
    def __init__(self, app):
        self.app = app
        self.st = st

    def run_streamlit_app(self):
        """Run the Streamlit web interface."""
        
        # Page config
        st.set_page_config(
            page_title="AI powered Document Analyzer",
            page_icon="📚",
            layout="wide"
        )

        # Initialize session state
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'current_query' not in st.session_state:
            st.session_state.current_query = None
        if 'agent_containers' not in st.session_state:
            st.session_state.agent_containers = []

        # Title
        st.title("📚 AI powered Document Analyzer")
        st.markdown("Query your documents and view relevant text, images, and tables.")

        # Sidebar for settings
        with st.sidebar:
            # PDF Upload Section
            st.header("📤 Upload PDF")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to add to the knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Uploaded PDFs", type="primary"):
                    self._process_uploaded_pdfs(uploaded_files)
            
            st.divider()
            
            # Agent Configuration
            st.header("⚙️ Settings")
            use_memory = st.checkbox("Enable Conversation Memory", value=True, 
                                    help="Agent remembers previous questions and answers")
            
            if st.button("🗑️ Clear Memory", help="Clear conversation history"):
                from agent import clear_memory
                clear_memory()
                st.success("Memory cleared!")
                st.rerun()
            
            st.divider()
            
            # Image filtering setting
            import config
            enable_image_filtering = st.checkbox(
                "Enable Intelligent Image Filtering", 
                value=config.ENABLE_IMAGE_FILTERING, 
                help="Use LLM to filter images by relevance (reduces irrelevant images but uses more resources)"
            )
            # Update config dynamically
            config.ENABLE_IMAGE_FILTERING = enable_image_filtering
            
            st.divider()
            
            # Result Limits
            num_images = st.slider("Max image results", 1, 10, 3, 
                                  help="Maximum number of images to retrieve")
            
            st.divider()
            
            # Agent behavior settings
            st.subheader("🤖 Agent Behavior")
            show_thinking = st.checkbox("Show agent thinking process", value=True,
                                       help="Display detailed agent reasoning and tool calls")
            
            st.divider()
            
            # Performance info
            st.subheader("⚡ Performance")
            import config
            perf_info = f"""
**Current Settings:**
- Model: `{config.AGENT_MODEL}`
- Context: `{config.OLLAMA_NUM_CTX}`
- Image Filter: `{'ON' if config.ENABLE_IMAGE_FILTERING else 'OFF'}`
- Max Images: `{config.MAX_RELEVANT_IMAGES}`
- TOP_K: `{config.TOP_K}`
"""
            st.markdown(perf_info)
            
            if config.ENABLE_IMAGE_FILTERING:
                st.warning("⚠️ Image filtering is ON (slower but more relevant results)")
            else:
                st.success("⚡ Image filtering is OFF (faster queries)")
            
            st.divider()
            
            # Show existing PDF files
            st.subheader("📚 Processed PDFs")
            pdf_files = self.app.get_all_pdf_files()
            if pdf_files:
                for pdf_file in pdf_files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"📄 {pdf_file}")
                    with col2:
                        if st.button("🗑️", key=f"del_{pdf_file}", help=f"Delete {pdf_file}"):
                            pdf_path = os.path.join(self.app.pdf_files_dir, pdf_file)
                            if os.path.exists(pdf_path):
                                os.remove(pdf_path)
                                st.success(f"Deleted {pdf_file}")
                                st.info("⚠️ Note: Vector DB not updated. Rebuild if needed.")
                                st.rerun()
        # Main query interface
        query = st.text_input("🔍 Enter your question:", placeholder="e.g., list all documents")

        if st.button("Search", type="primary") or (query and st.session_state.get('trigger_search')):
            if query:
                # Only execute if query has changed
                if query != st.session_state.get('last_executed_query'):
                    st.session_state.current_query = query
                    st.session_state.last_executed_query = query
                    st.session_state.trigger_search = False
                    # Store in history
                    st.session_state.query_history.append((query, None))
                    
                    # Execute immediately without rerun
                    self._handle_agent_mode(query, use_memory=use_memory, 
                                           show_thinking=show_thinking)
        
        # Show cached results if available
        elif st.session_state.get('last_executed_query'):
            last_query = st.session_state.last_executed_query
            if last_query in st.session_state.get('cached_results', {}):
                cached = st.session_state.cached_results[last_query]
                self._display_cached_results(cached)

        # Show query history
        if st.session_state.query_history:
            with st.sidebar:
                st.divider()
                st.subheader("📜 Query History")
                for i, (q, _) in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                    st.text(f"{i}. {q[:40]}...")


    def _process_uploaded_pdfs(self, uploaded_files):
        """Process and add uploaded PDFs to vector database."""
        with st.spinner("Processing PDFs..."):
            success_count = 0
            for uploaded_file in uploaded_files:
                pdf_save_path = os.path.join(self.app.pdf_files_dir, uploaded_file.name)
                
                if os.path.exists(pdf_save_path):
                    st.warning(f"⚠️ {uploaded_file.name} already exists. Skipping...")
                    continue
                
                with open(pdf_save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    num_texts, num_images, num_tables = self.app.add_pdf_to_vector_db(pdf_save_path)
                    st.success(f"✅ {uploaded_file.name} processed: {num_texts} texts, {num_images} images, {num_tables} tables")
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    if os.path.exists(pdf_save_path):
                        os.remove(pdf_save_path)
            
            if success_count > 0:
                # Reset caches after updating vector DB
                reset_streamlit_caches()
                st.success(f"🎉 Successfully processed {success_count} PDF(s)!")
                st.rerun()

    def _handle_agent_mode(self, query, use_memory=True, show_thinking=True):
        """Handle agent mode query execution."""
        import time
        
        try:
            # Check cache first
            cache_key = f"{query}_{use_memory}_{show_thinking}"
            if 'cached_results' not in st.session_state:
                st.session_state.cached_results = {}
            
            # Use a unique key based on the query to ensure fresh containers
            query_id = f"query_{len(st.session_state.query_history)}"
            
            st.divider()
            if show_thinking:
                st.subheader("🤖 Agent Thinking Process")
            
            from agent import stream_agent
            
            agent_state = {
                'step_count': 0,
                'final_answer': None,
                'image_results': [],
                'table_results': [],
                'current_tool': None,
                'query_id': query_id,
                'pending_tools': []
            }
            
            status_placeholder = st.empty()
            status_placeholder.info("🔄 Agent starting...")
            
            # Start timing
            start_time = time.time()
            
            # Create a fresh container for this query
            if show_thinking:
                thinking_container = st.container()
                with thinking_container:
                    self._stream_agent_events(stream_agent(query, use_memory=use_memory), agent_state, status_placeholder, show_thinking)
            else:
                self._stream_agent_events(stream_agent(query, use_memory=use_memory), agent_state, status_placeholder, show_thinking)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            self._display_agent_results(agent_state, status_placeholder)
            
            # Show performance info
            st.caption(f"⏱️ Query completed in {elapsed_time:.2f} seconds ({agent_state['step_count']} steps)")
            
            # Cache results
            st.session_state.cached_results[cache_key] = agent_state
            
        except Exception as e:
            st.error(f"❌ Agent error: {str(e)}")
            st.exception(e)
            st.info("💡 Tip: Make sure Ollama is running and documents are in the database.")

    def _stream_agent_events(self, event_stream, state, status_placeholder, show_thinking=True):
        """Stream and display agent events."""
        query_id = state.get('query_id', 'default')
        
        try:
            for event in event_stream:
                if event['type'] == 'tool_call':
                    state['step_count'] += 1
                    state['current_tool'] = event['tool']
                    status_placeholder.info(f"🔄 Step {state['step_count']}: Using tool '{event['tool']}'...")
                    state.setdefault('pending_tools', []).append({
                        'name': event['tool'],
                        'input': event.get('input', {}),
                        'id': event.get('id')
                    })
                    
                    if show_thinking:
                        with st.expander(f"🔧 Step {state['step_count']}: Tool **{event['tool']}**", expanded=True):
                            st.write("**Parameters:**")
                            st.json(event['input'])
                
                elif event['type'] == 'tool_result':
                    if show_thinking:
                        self._handle_tool_result(event, state, query_id)
                
                elif event['type'] == 'answer':
                    state['final_answer'] = event['content']
                    status_placeholder.success("✅ Agent generated answer!")
            
            if state['step_count'] == 0 and not state['final_answer']:
                status_placeholder.warning("⚠️ Agent didn't execute any steps.")
        
        except Exception as stream_error:
            status_placeholder.error(f"❌ Streaming error: {str(stream_error)}")
            st.exception(stream_error)

    def _handle_tool_result(self, event, state, query_id):
        """Process and display tool result."""
        raw_content = event.get('content')
        content_text = self._stringify_tool_content(raw_content)
        result_preview = content_text[:300] + ("..." if len(content_text) > 300 else "")
        tool_call = self._resolve_tool_call_context(state, event.get('tool_call_id'))
        tool_name = tool_call.get('name') if tool_call else state.get('current_tool')
        
        # Track results by tool type (standard_search now includes images/tables)
        if tool_name == 'standard_search' and 'IMAGES ===' in content_text:
            state['image_results'].append(raw_content)
        if tool_name == 'standard_search' and 'TABLES ===' in content_text:
            state['table_results'].append(raw_content)
        
        with st.status(f"✅ Result from step {state['step_count']}", expanded=False):
            st.text(result_preview)
            if len(content_text) > 300:
                with st.expander("Show full result"):
                    st.code(content_text, language="text")

    def _display_agent_results(self, state, status_placeholder):
        """Display final agent results."""
        st.divider()
        st.subheader("📝 Final Answer")
        
        if state['final_answer']:
            st.success(state['final_answer'])
        else:
            st.warning("⚠️ No answer received from agent.")
            st.info(f"The agent executed {state['step_count']} steps but didn't generate a final answer.")
        
        if state['image_results']:
            st.divider()
            st.subheader("🖼️ Found Images")
            self._display_images_from_tool_results(state['image_results'])


    def _display_images_from_tool_results(self, image_tool_results):
        """Parse tool results and display images found by the agent."""
        parsed_entries = []
        for payload in image_tool_results:
            parsed_entries.extend(self._parse_agent_image_results(payload))
        
        valid_entries = [entry for entry in parsed_entries if entry.get('path')]
        if not valid_entries:
            st.warning("⚠️ Agent found images but could not extract file paths.")
            return
        
        image_paths = [entry['path'].strip() for entry in valid_entries]
        metadata = [
            {
                'source': entry.get('source'),
                'page': entry.get('page'),
                'caption': entry.get('caption')
            }
            for entry in valid_entries
        ]
        
        self._display_image_grid(image_paths, metadata)
    
    
    def _display_tables_from_tool_results(self, table_tool_results):
        """Parse tool results and display tables found by the agent."""
        parsed_tables = []
        for payload in table_tool_results:
            parsed_tables.extend(self._parse_agent_table_results_json(payload))
        
        if not parsed_tables:
            st.warning("⚠️ Agent returned tables but could not extract table data.")
            return

        for idx, table_info in enumerate(parsed_tables, 1):
            source = table_info.get('source', 'Unknown')
            page = table_info.get('page', '?')
            preview = table_info.get('preview', '')
            st.markdown(f"**📊 Table {idx} - {source} (Page {page})**")

            df = self._table_preview_to_dataframe(preview)
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.code(preview or "No preview available", language="text")

            if preview and preview.endswith('...'):
                st.caption("Preview truncated to first 300 characters.")

    def _display_image_grid(self, image_paths, metadata=None):
        """Display images in a grid layout."""
        num_cols = min(len(image_paths), 3)
        cols = st.columns(num_cols)
        
        for i, img_path in enumerate(image_paths):
            with cols[i % num_cols]:
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path)
                        
                        if metadata:
                            meta = metadata[i]
                            caption = f"{meta.get('source', 'Unknown')} - Page {meta.get('page')}"
                        else:
                            filename = os.path.basename(img_path)
                            page_match = re.search(r'_p(\d+)_', filename)
                            page = page_match.group(1) if page_match else "?"
                            caption = f"Page {page}"
                        
                        st.image(image, caption=caption, use_container_width=True)
                        
                        with st.expander("Details"):
                            if metadata:
                                meta = metadata[i]
                                st.write(f"**File:** {meta.get('source', 'Unknown')}")
                                st.write(f"**Page:** {meta.get('page')}")
                                st.write(f"**Caption:** {meta.get('caption', 'No caption')}")
                            st.write(f"**Path:** `{img_path}`")
                            st.write(f"**Size:** {image.size[0]}x{image.size[1]}px")
                    except Exception as e:
                        st.error(f"Could not load image: {e}")
                else:
                    st.warning(f"Image not found: {img_path}")

    def _display_image_results(self, image_metas):
        """Display image search results."""
        st.subheader(f"Found {len(image_metas)} relevant images")
        image_paths = [img.get('path') for img in image_metas if img.get('path')]
        self._display_image_grid(image_paths, image_metas)


    def _display_table_results(self, table_metas):
        """Display table search results."""
        st.subheader(f"Found {len(table_metas)} relevant tables")
        
        for i, tb in enumerate(table_metas, 1):
            st.markdown(f"### 📊 Table {i} - {tb.get('source', 'Unknown')} (Page {tb.get('page')})")
            
            try:
                table_data = json.loads(tb.get('json', '[]')) if isinstance(tb.get('json', '[]'), str) else tb.get('json', [])
                
                if not table_data:
                    st.warning("⚠️ This table has no data (empty array from PDF extraction)")
                    st.text(tb.get('preview', 'No preview available'))
                else:
                    df = self._create_dataframe_from_table(table_data)
                    if df is not None:
                        st.dataframe(df, use_container_width=True)
                        with st.expander("Show raw preview text"):
                            st.text(tb.get('preview', 'No preview'))
                    else:
                        st.warning("⚠️ Table contains only empty cells")
                        st.text(tb.get('preview', 'No preview'))
            except Exception as e:
                st.error(f"Could not format table: {e}")
                st.text(tb.get('preview', 'No preview'))
            
            st.divider()

    def _create_dataframe_from_table(self, table_data):
        """Create a pandas DataFrame from table data with proper headers."""
        non_empty_rows = [row for row in table_data if any(cell for cell in row)]
        
        if not non_empty_rows:
            return None
        
        if len(table_data) > 1:
            headers = self._clean_headers(table_data[0])
            return pd.DataFrame(table_data[1:], columns=headers)
        
        return pd.DataFrame(table_data)

    def _clean_headers(self, headers):
        """Clean and deduplicate table headers."""
        cleaned = [
            f"Column_{idx}" if h is None or str(h).strip() == '' else str(h).strip()
            for idx, h in enumerate(headers)
        ]
        
        seen = {}
        unique = []
        for h in cleaned:
            if h in seen:
                seen[h] += 1
                unique.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique.append(h)
        
        return unique

    def _resolve_tool_call_context(self, state, tool_call_id):
        """Match tool results to their originating tool calls."""
        queue = state.get('pending_tools', [])
        if not queue:
            return None
        if tool_call_id:
            for idx, record in enumerate(queue):
                if record.get('id') == tool_call_id:
                    return queue.pop(idx)
        return queue.pop(0)

    def _parse_agent_table_results(self, table_tool_results):
        """Extract structured table info from agent tool output strings (text format)."""
        # New format: [Table 1] Source: ..., Page ...\nPreview: ...
        pattern_new = re.compile(
            r"\[Table\s+(\d+)\]\s*Source:\s*(.*?),\s*Page\s*([^\n]+?)\s*\nPreview:\s*([\s\S]*?)(?=\n\[Table\s+\d+\]|\Z)",
            re.MULTILINE
        )
        # Legacy format: [1] Source: ..., Page ...\nPreview: ...
        pattern_legacy = re.compile(
            r"\[(\d+)\]\s*Source:\s*(.*?),\s*Page\s*([^\n]+?)\s*\nPreview:\s*\n?([\s\S]*?)(?=\n\[\d+\]|\Z)",
            re.MULTILINE
        )
        
        tables = []
        for raw in table_tool_results:
            if not raw:
                continue
            raw_text = raw.strip()
            
            # Try new format first
            matches = list(pattern_new.finditer(raw_text))
            if matches:
                for match in matches:
                    tables.append({
                        'index': int(match.group(1)),
                        'source': match.group(2).strip(),
                        'page': match.group(3).strip(),
                        'preview': match.group(4).strip()
                    })
            else:
                # Fallback to legacy format
                for match in pattern_legacy.finditer(raw_text):
                    tables.append({
                        'index': int(match.group(1)),
                        'source': match.group(2).strip(),
                        'page': match.group(3).strip(),
                        'preview': match.group(4).strip()
                    })
        return tables
    
    def _parse_agent_table_results_json(self, payload):
        """Parse agent table tool output (JSON format, similar to images)."""
        if payload is None:
            return []
        
        # Handle list of dicts (message chunks)
        if isinstance(payload, list):
            if all(isinstance(item, dict) for item in payload):
                textual_parts = [
                    item.get('text') or item.get('content')
                    for item in payload
                    if isinstance(item, dict) and ('text' in item or 'content' in item)
                ]
                combined_text = "\n".join(part for part in textual_parts if part)
                if combined_text:
                    return self._parse_agent_table_results_json(combined_text)
            if all(isinstance(item, str) for item in payload):
                combined_text = "\n".join(payload)
                return self._parse_agent_table_results_json(combined_text)
            return []
        
        # Handle dict wrappers
        if isinstance(payload, dict):
            if 'text' in payload:
                return self._parse_agent_table_results_json(payload['text'])
            if 'content' in payload:
                return self._parse_agent_table_results_json(payload['content'])
            if 'items' in payload:
                return self._normalize_table_result_payload(payload)
            return []
        
        # Handle string (JSON or plain text)
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            try:
                structured = json.loads(text)
                return self._normalize_table_result_payload(structured)
            except json.JSONDecodeError:
                # Fallback to legacy text parsing
                return self._parse_agent_table_results([text])
        
        return []
    
    def _normalize_table_result_payload(self, payload):
        """Normalize structured table payload into list of dicts."""
        if isinstance(payload, dict):
            # Check for error or empty result
            if 'error' in payload:
                st.error(f"Table search error: {payload['error']}")
                return []
            if 'message' in payload and not payload.get('items'):
                st.info(payload['message'])
                return []
            if 'items' in payload:
                payload = payload['items']
            else:
                payload = [payload]
        
        if not isinstance(payload, list):
            return []
        
        normalized = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            normalized.append({
                'source': item.get('source', 'Unknown'),
                'page': item.get('page', 'Unknown'),
                'preview': item.get('preview', 'No preview'),
                'json_data': item.get('json_data')
            })
        return normalized

    def _table_preview_to_dataframe(self, preview_text):
        """Convert table preview text into a pandas DataFrame when possible."""
        if not preview_text:
            return None
        lines = [line for line in preview_text.splitlines() if line.strip()]
        if not lines:
            return None

        rows = []
        for line in lines:
            stripped = line.strip()
            if set(stripped) <= {'-', '='}:
                continue
            if '|' in stripped:
                segments = [cell.strip() for cell in stripped.strip('|').split('|')]
            else:
                segments = [cell.strip() for cell in re.split(r'\s{2,}', stripped)]
            segments = [cell for cell in segments if cell]
            if segments:
                rows.append(segments)

        if len(rows) < 2:
            return None

        max_cols = max(len(row) for row in rows)
        normalized = [row + [''] * (max_cols - len(row)) for row in rows]
        headers = normalized[0]
        data = normalized[1:]

        try:
            return pd.DataFrame(data, columns=headers)
        except Exception:
            return None

    def _stringify_tool_content(self, content):
        """Convert tool output payloads (str/list/dict) into plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [self._stringify_tool_content(item) for item in content]
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if 'text' in content:
                return str(content['text'])
            if 'content' in content:
                return self._stringify_tool_content(content['content'])
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _parse_agent_image_results(self, payload):
        """Parse agent tool output for image metadata and paths."""
        if payload is None:
            return []
        if isinstance(payload, list):
            # Lists coming from ToolMessage often wrap text chunks
            if all(isinstance(item, dict) for item in payload):
                # Try to extract structured entries directly first
                structured_entries = self._normalize_image_result_payload(payload)
                if structured_entries:
                    return structured_entries
                textual_parts = [
                    item.get('text') or item.get('content')
                    for item in payload
                    if isinstance(item, dict) and ('text' in item or 'content' in item)
                ]
                combined_text = "\n".join(part for part in textual_parts if part)
                if combined_text:
                    return self._parse_agent_image_results(combined_text)
            # List of strings
            if all(isinstance(item, str) for item in payload):
                combined_text = "\n".join(payload)
                return self._parse_agent_image_results(combined_text)
            # Fallback
            return []
        if isinstance(payload, dict):
            structured = self._normalize_image_result_payload(payload)
            if structured:
                return structured
            if 'text' in payload:
                return self._parse_agent_image_results(payload['text'])
            if 'content' in payload:
                return self._parse_agent_image_results(payload['content'])
            return []
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return []
            try:
                structured = json.loads(text)
                return self._normalize_image_result_payload(structured)
            except json.JSONDecodeError:
                return self._parse_image_text_block(text)
        return []

    def _normalize_image_result_payload(self, payload):
        """Normalize different structured payload shapes into a list of dicts."""
        if isinstance(payload, dict) and 'items' in payload:
            payload = payload['items']
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return []
        normalized = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            path = item.get('path')
            source = item.get('source')
            caption = item.get('caption')
            page = item.get('page')
            # Require at least a path to treat as structured image info
            if not path:
                continue
            normalized.append({
                'path': path,
                'source': source or 'Unknown',
                'page': page or 'Unknown',
                'caption': caption or 'No description'
            })
        return normalized

    def _parse_image_text_block(self, text_block):
        """Fallback regex parser for text-format image tool output."""
        # New format: [Image 1] ✓ Source: ..., Page ...\nCaption: ...\nPath: ...
        pattern_new = re.compile(
            r"\[Image\s+(\d+)\]\s*[✓✗]?\s*Source:\s*(.*?),\s*Page\s*([^\n]+)\s*\nCaption:\s*([^\n]+)\s*\nPath:\s*([^\n]+)",
            re.MULTILINE
        )
        matches = list(pattern_new.finditer(text_block))
        
        # Legacy format: [1] ✓ Source: ..., Page ...\nPath: ...\nDescription: ...
        if not matches:
            pattern_legacy = re.compile(
                r"\[(\d+)\]\s*[✓✗]?\s*Source:\s*(.*?),\s*Page\s*([^\n]+)\s*\nPath:\s*([^\n]+)\s*\nDescription:\s*([\s\S]*?)(?=\n\[\d+\]|\Z)",
                re.MULTILINE
            )
            matches = pattern_legacy.finditer(text_block)
            entries = []
            for match in matches:
                entries.append({
                    'path': match.group(4).strip(),
                    'source': match.group(2).strip(),
                    'page': match.group(3).strip(),
                    'caption': match.group(5).strip()
                })
            return entries
        
        # Parse new format
        entries = []
        for match in matches:
            entries.append({
                'path': match.group(5).strip(),
                'source': match.group(2).strip(),
                'page': match.group(3).strip(),
                'caption': match.group(4).strip()
            })
        return entries
    
    def _display_cached_results(self, agent_state):
        """Display previously cached agent results without re-execution."""
        st.divider()
        st.info("📦 Showing cached results")
        
        st.divider()
        st.subheader("📝 Final Answer")
        
        if agent_state.get('final_answer'):
            st.success(agent_state['final_answer'])
        else:
            st.warning("⚠️ No answer in cached results.")
        
        if agent_state.get('image_results'):
            st.divider()
            st.subheader("🖼️ Found Images")
            self._display_images_from_tool_results(agent_state['image_results'])