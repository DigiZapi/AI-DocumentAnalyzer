import streamlit as st
import os
import json
import pandas as pd
from PIL import Image
import sys
from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from rag_backend import build_vector_db, retriever
from utility.caption_images import caption_images
from utility.pdf_reader import PDFReader
from agent import query_with_agent
from streamlit_webinterface import StreamlitWebInterface

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
# LLM functionality moved to agent - no longer needed in app




class DocumentAnalyserApp:
    """Main application class for Document Analyser."""

    def __init__(self, ):
        
        # Set up paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.pdf_files_dir = os.path.join(self.project_root, "pdf_files")
        self.image_output_dir = os.path.join(self.project_root, "extracted_images")
        
        # Create directories if they don't exist
        os.makedirs(self.pdf_files_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        # init streamlit
        self.st = StreamlitWebInterface(self)

        self.template = """
        You are a helpful assistant that provides clear EXPLANATIONS and step-by-step INSTRUCTIONS based **only** on the context below.
        
        IMPORTANT GUIDELINES:
        - For "how to" questions: Provide numbered step-by-step instructions
        - For "explain" questions: Explain the concept in detail
        - DON'T just describe what's in images or tables - EXPLAIN how to use that information
        - Reference specific UI elements, menus, buttons, or settings shown in screenshots
        - If you see BIOS/settings screens, tell users exactly which options to select
        - Combine information from text, images, and tables to give complete instructions
        - If the context doesn't contain the answer, say you don't know
        - Cite sources: mention page numbers and whether info came from text, images, or tables

        Context:
        {context}

        Question:
        {question}

        Answer (provide clear instructions or explanations):
        """


    def read_and_process_pdf(self, pdf_path):
        """Read PDF, extract texts, images, tables, and generate captions."""
        pdf_reader = PDFReader(pdf_path, self.image_output_dir)
        text_chunks, image_data, table_data = pdf_reader.extract_pdf()

        print("Text chunks:", len(text_chunks))
        print("Images:", len(image_data))
        print("Tables:", len(table_data))

        # create captions for images
        image_captions = caption_images(image_data)
        print("Image captions:", image_captions)

        return text_chunks, image_data, table_data, image_captions
    

    def create_new_vector_db(self, text_chunks, image_data, table_data, image_captions):
        
        """Create a new vector database from extracted data."""
        build_vector_db(text_chunks, image_data, table_data, image_captions)
        print("Vector DB created successfully.")
    
    
    def add_pdf_to_vector_db(self, pdf_path):
        """Process a single PDF and add it to the existing vector database."""
        print(f"\n📄 Processing PDF: {pdf_path}")
        
        # Extract content from PDF
        text_chunks, image_data, table_data, image_captions = self.read_and_process_pdf(pdf_path)
        
        # Add to existing vector DB (incremental update)
        build_vector_db(text_chunks, image_data, table_data, image_captions)
        
        return len(text_chunks), len(image_data), len(table_data)
    
    
    def get_all_pdf_files(self):
        """Get list of all PDF files in the pdf_files directory."""
        if not os.path.exists(self.pdf_files_dir):
            return []
        
        pdf_files = [f for f in os.listdir(self.pdf_files_dir) if f.lower().endswith('.pdf')]
        return sorted(pdf_files)
    
    
    def query_with_agent(self, query):
        """Query using agent - the agent decides which tools to use."""
        return query_with_agent(query)


    def run_streamlit_app(self):
        """Run the Streamlit web interface."""
        self.st.run_streamlit_app()



# Main entry point with menu
def main():
    """Main entry point with CLI menu."""
    import sys
    
    # Check if running via streamlit by checking if we're in a streamlit runtime context
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            # We're inside a Streamlit runtime, run the app
            app = DocumentAnalyserApp()
            app.run_streamlit_app()
            return
    except (ImportError, AttributeError):
        pass
    
    # Also check command line args for streamlit
    if 'streamlit' in sys.argv[0] or any('streamlit' in str(arg).lower() for arg in sys.argv):
        app = DocumentAnalyserApp()
        app.run_streamlit_app()
        return
    
    print("\n" + "="*60)
    print("  📚 Document Analyser - Main Menu")
    print("="*60)
    print("\nChoose an option:")
    print("  1) Create/Rebuild Vector Database")
    print("  2) Run Streamlit Web Interface")
    print("  3) Exit")
    print("-"*60)
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🔧 Creating Vector Database...")
        print("-"*60)
        app = DocumentAnalyserApp()
        
        # Get all PDFs from pdf_files directory
        pdf_files = app.get_all_pdf_files()
        
        if not pdf_files:
            print("\n⚠️  No PDF files found in pdf_files directory!")
            print(f"   Please add PDF files to: {app.pdf_files_dir}")
            print("   Or use the web interface (option 2) to upload PDFs.\n")
            print("="*60 + "\n")
            return
        
        print(f"\n📚 Found {len(pdf_files)} PDF file(s) to process:")
        for pdf_file in pdf_files:
            print(f"   • {pdf_file}")
        
        print(f"\n💾 Extracting images to: {app.image_output_dir}\n")
        
        # Process all PDFs
        all_text_chunks = []
        all_image_data = []
        all_table_data = []
        all_image_captions = {}
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(app.pdf_files_dir, pdf_file)
            print(f"\n📄 Processing: {pdf_file}")
            
            text_chunks, image_data, table_data, image_captions = app.read_and_process_pdf(pdf_path)
            
            all_text_chunks.extend(text_chunks)
            all_image_data.extend(image_data)
            all_table_data.extend(table_data)
            all_image_captions.update(image_captions)
        
        print("\n📊 Total Extraction Summary:")
        print(f"  • Text chunks: {len(all_text_chunks)}")
        print(f"  • Images: {len(all_image_data)}")
        print(f"  • Tables: {len(all_table_data)}")
        print(f"  • Image captions: {len(all_image_captions)}")
        
        print("\n🔨 Building vector database...")
        app.create_new_vector_db(all_text_chunks, all_image_data, all_table_data, all_image_captions)
        
        print("\n✅ Vector database created successfully!")
        print("="*60 + "\n")
        
    elif choice == "2":
        print("\n🌐 Starting Streamlit Web Interface...")
        print("-"*60)
        
        # Try to launch streamlit with the correct venv
        try:
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            venv_streamlit = os.path.join(project_root, ".venv_da", "bin", "streamlit")
            
            if os.path.exists(venv_streamlit):
                print(f"Launching Streamlit from: {venv_streamlit}\n")
                subprocess.run([venv_streamlit, "run", __file__])
            else:
                print("\n⚠️  Virtual environment not found!")
                print(f"   Expected: {venv_streamlit}")
                print("\n   Please run one of these commands:")
                print("   • source .venv_da/bin/activate && streamlit run src/app.py")
                print("   • ./start.sh and select option 4\n")
                print("="*60 + "\n")
        except Exception as e:
            print(f"\n❌ Could not launch Streamlit: {e}")
            print("\n   Please run manually:")
            print("   • source .venv_da/bin/activate && streamlit run src/app.py")
            print("   • Or use: ./start.sh and select option 4\n")
            print("="*60 + "\n")
        
    elif choice == "3":
        print("\n👋 Goodbye!\n")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please run again and select 1, 2, or 3.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
    