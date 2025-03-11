import streamlit as st
import os
import tempfile
import subprocess
import base64
import json
import re
import httpx
from pathlib import Path
from openai import OpenAI, AzureOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Set page configuration
st.set_page_config(
    page_title="AI PDF Document Generator",
    page_icon="📄",
    layout="wide"
)

# App title and description
st.title("AI PDF Document Generator")
st.markdown("""
This application generates professional documents in PDF format using AI.
Select your document type, provide the necessary details, and let AI create a formatted document for you.
You can also chat with the AI to refine your document iteratively.
""")

# Initialize session state
if 'provider' not in st.session_state:
    st.session_state.provider = "openai"
if 'client' not in st.session_state:
    st.session_state.client = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'latex_content' not in st.session_state:
    st.session_state.latex_content = None
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None
if 'document_type' not in st.session_state:
    st.session_state.document_type = None
if 'details' not in st.session_state:
    st.session_state.details = None
if 'additional_instructions' not in st.session_state:
    st.session_state.additional_instructions = None

# List of models with beta structured outputs support
BETA_STRUCTURED_MODELS = [
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
]

# Document types available in the application
DOCUMENT_TYPES = [
    "Consent Form",
    "Contract",
    "Agreement",
    "Legal Notice",
    "Business Proposal",
    "Cover Letter",
    "Invoice",
    "Report",
    "Certificate",
    "Memorandum",
    "Press Release",
    "Custom Document"
]


# Define Pydantic models for structured outputs
class LatexDocument(BaseModel):
    latex_content: str = Field(description=" that can be comComplete LaTeX document contentpiled to PDF")


class DocumentFeedback(BaseModel):
    updated_latex: str = Field(description="Updated LaTeX document content with the requested changes")
    summary_of_changes: str = Field(description="Summary of changes made to the document")


# Helper Functions
def supports_beta_structured_outputs(model_name):
    """Check if the model supports beta structured outputs."""
    for supported_model in BETA_STRUCTURED_MODELS:
        if supported_model in model_name.lower():
            return True
    return False


def clean_latex_content(content):
    """Clean LaTeX content to ensure it's valid for compilation."""
    # Extract content from code blocks if present
    if "```latex" in content:
        start_marker = "```latex"
        end_marker = "```"
        start_idx = content.find(start_marker) + len(start_marker)
        end_idx = content.rfind(end_marker)
        if start_idx > -1 and end_idx > start_idx:
            content = content[start_idx:end_idx].strip()

    # Ensure content starts with documentclass
    if not content.strip().startswith("\\documentclass"):
        doc_class_idx = content.find("\\documentclass")
        if doc_class_idx > 0:
            content = content[doc_class_idx:].strip()

    # Ensure content ends with end{document}
    end_doc_idx = content.find("\\end{document}")
    if end_doc_idx > 0:
        content = content[:end_doc_idx + len("\\end{document}")].strip()

    return content


# AI Provider Setup
def initialize_ai_provider():
    """Setup AI provider (OpenAI or Azure OpenAI)."""
    st.sidebar.title("AI Provider Settings")

    # Provider selection
    provider = st.sidebar.radio(
        "Select AI Provider",
        options=["OpenAI", "Azure OpenAI"],
        horizontal=True
    )

    st.session_state.provider = provider.lower().replace(" ", "_")

    if st.session_state.provider == "openai":
        # Standard OpenAI settings
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        model = st.sidebar.selectbox(
            "Model",
            options=["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
            index=2  # Default to gpt-4o
        )

        if api_key:
            try:
                st.session_state.client = OpenAI(api_key=api_key)
                st.session_state.model = model
                return True
            except Exception as e:
                st.sidebar.error(f"Error initializing OpenAI client: {str(e)}")
                return False
        else:
            st.sidebar.warning("Please enter your OpenAI API key.")
            return False
    else:
        # Azure OpenAI settings
        col1, col2 = st.sidebar.columns(2)

        with col1:
            api_key = st.text_input("Azure API Key", type="password")

        with col2:
            api_endpoint = st.text_input("Azure Endpoint URL")

        deployment_name = st.sidebar.text_input("Model Deployment Name")
        api_version = st.sidebar.text_input("API Version", value="2023-05-15")

        if api_key and api_endpoint and deployment_name:
            try:
                st.session_state.client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=api_endpoint,
                    http_client=httpx.Client(verify=False)
                )
                st.session_state.model = deployment_name
                return True
            except Exception as e:
                st.sidebar.error(f"Error initializing Azure OpenAI client: {str(e)}")
                return False
        else:
            st.sidebar.warning("Please enter your Azure OpenAI credentials.")
            return False


# Document Generation
def generate_document(document_type, details, additional_instructions):
    """Generate latex document content using the selected AI provider."""
    if not st.session_state.client or not st.session_state.model:
        st.error("AI provider credentials are not set.")
        return None

    client = st.session_state.client
    model = st.session_state.model
    provider = st.session_state.provider

    # Check if we can use beta structured outputs (only for OpenAI)
    use_beta_structured = supports_beta_structured_outputs(model) and provider == "openai"

    if use_beta_structured:
        prompt = f"""
        Generate a professional {document_type} with the following details:

        {details}

        Additional instructions: {additional_instructions}

        The latex_content should be a complete LaTeX document that can be compiled directly into a PDF.
        Include all necessary LaTeX packages and proper structure.
        Make sure the document is well-organized, professional, and ready for use.
        """

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a professional document creation assistant specializing in LaTeX."},
                    {"role": "user", "content": prompt}
                ],
                response_format=LatexDocument,
                temperature=0.2
            )

            # Get the parsed response
            return response.choices[0].message.parsed.latex_content

        except Exception as e:
            st.error(f"Error generating document with beta structured outputs: {str(e)}")
            use_beta_structured = False

    if not use_beta_structured:
        prompt = f"""
        Generate a professional {document_type} with the following details:

        {details}

        Additional instructions: {additional_instructions}

        Format it as a complete LaTeX document that can be compiled directly into a PDF.
        Include all necessary LaTeX packages and proper structure.
        Make sure the document is well-organized, professional, and ready for use.
        IMPORTANT: Return ONLY valid LaTeX code with no explanatory text before or after.
        Start with \\documentclass and end with \\end{{document}}.
        """

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional document creation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating document: {str(e)}")
            return None


def update_document(feedback):
    if not st.session_state.client or not st.session_state.model:
        st.error("AI provider credentials are not set.")
        return None, None

    client = st.session_state.client
    model = st.session_state.model
    provider = st.session_state.provider

    if not st.session_state.latex_content:
        st.error("No document has been generated yet.")
        return None, None

    messages = [
        {"role": "system",
         "content": "You are a professional document creation assistant specializing in LaTeX. You help users refine their documents based on feedback."}
    ]

    messages.append({
        "role": "user",
        "content": f"""
        I need a professional {st.session_state.document_type} with these details:

        {st.session_state.details}

        Additional instructions: {st.session_state.additional_instructions}
        """
    })

    # Add the initial response with LaTeX content
    messages.append({
        "role": "assistant",
        "content": f"I've created the document as requested. Here's the LaTeX source code:\n\n```latex\n{st.session_state.latex_content}\n```"
    })

    # Add chat history
    for msg in st.session_state.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the latest feedback
    messages.append({"role": "user", "content": f"Please update the document with the following changes: {feedback}"})

    # Check if we can use beta structured outputs (only for OpenAI)
    use_beta_structured = supports_beta_structured_outputs(model) and provider == "openai"

    if use_beta_structured:
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=DocumentFeedback,
                temperature=0.2
            )

            # Get the parsed response
            updated_latex = response.choices[0].message.parsed.updated_latex
            summary = response.choices[0].message.parsed.summary_of_changes

            return updated_latex, summary

        except Exception as e:
            st.error(f"Error updating document with beta structured outputs: {str(e)}")
            # Fallback to traditional approach
            use_beta_structured = False

    # Traditional approach (for Azure or as fallback)
    if not use_beta_structured:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4000,
                temperature=0.2
            )

            content = response.choices[0].message.content

            # Extract LaTeX code from response if it's wrapped in code blocks
            if "```latex" in content:
                latex_content = re.search(r"```latex\n([\s\S]*?)\n```", content)
                if latex_content:
                    updated_latex = latex_content.group(1)
                else:
                    updated_latex = clean_latex_content(content)
            else:
                updated_latex = clean_latex_content(content)

            # Try to extract a summary from the response
            summary = "Document updated with your requested changes."
            if "I've updated" in content or "I have updated" in content or "Changes made" in content:
                # Try to extract a summary paragraph
                summary_match = re.search(r"(?:I've updated|I have updated|Changes made)(.*?)(?:```|$)", content,
                                          re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip()

            return updated_latex, summary

        except Exception as e:
            st.error(f"Error updating document: {str(e)}")
            return None, None


# Document Processing
def compile_latex_to_pdf(latex_content):
    """Compile LaTeX content to PDF."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary directory for LaTeX compilation
        temp_dir = Path(tmpdir)
        latex_file = temp_dir / "document.tex"

        # Write the LaTeX content to a file
        with open(latex_file, "w") as f:
            f.write(latex_content)

        # Compile the LaTeX file to PDF
        try:
            # Run pdflatex twice to resolve references
            process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(temp_dir), str(latex_file)],
                capture_output=True,
                text=True
            )

            # Check if compilation was successful
            if process.returncode != 0:
                st.error("LaTeX compilation failed. Please check your document for errors.")
                st.code(process.stderr, language="text")
                return None

            # Run pdflatex again to resolve references
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(temp_dir), str(latex_file)],
                capture_output=True
            )

            # Read the generated PDF
            pdf_file = temp_dir / "document.pdf"
            if pdf_file.exists():
                with open(pdf_file, "rb") as f:
                    pdf_content = f.read()
                return pdf_content
            else:
                st.error("PDF generation failed.")
                return None
        except Exception as e:
            st.error(f"Error compiling LaTeX: {str(e)}")
            return None


def display_pdf(pdf_content):
    """Display PDF in the Streamlit app and provide download option."""
    # Encode PDF in base64
    b64_pdf = base64.b64encode(pdf_content).decode("utf-8")

    # Display PDF
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# Main Application Function
def main():
    # Setup AI provider in sidebar
    api_initialized = initialize_ai_provider()

    # Create two columns: one for document setup, one for preview/chat
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Document Setup")

        # Document type selection
        selected_type = st.selectbox("Select Document Type", DOCUMENT_TYPES)

        if selected_type == "Custom Document":
            custom_type = st.text_input("Enter Custom Document Type")
            document_type = custom_type if custom_type else "Document"
        else:
            document_type = selected_type

        # Document details
        st.subheader("Document Details")
        st.markdown("Provide the necessary information for your document. Be specific to get the best results.")

        # Text area for document details
        details = st.text_area(
            "Document Details",
            height=200,
            placeholder="Enter all relevant information for your document here.\n\nExample:\n- Names of parties involved\n- Dates\n- Locations\n- Terms and conditions\n- Financial details\n- Other specific information"
        )

        # Additional instructions
        additional_instructions = st.text_area(
            "Additional Instructions (Optional)",
            height=100,
            placeholder="Enter any specific formatting requests, tone preferences, or additional instructions for the AI."
        )

        # Generate document button
        if st.button("Generate Document") and api_initialized:
            if not details:
                st.warning("Please enter document details.")
                return

            # Store document details in session state
            st.session_state.document_type = document_type
            st.session_state.details = details
            st.session_state.additional_instructions = additional_instructions

            # Clear chat history for new document
            st.session_state.chat_history = []

            # Generate document
            with st.spinner("Generating LaTeX document content..."):
                latex_content = generate_document(document_type, details, additional_instructions)

            if latex_content:
                # Clean the LaTeX content
                cleaned_latex = clean_latex_content(latex_content)
                st.session_state.latex_content = cleaned_latex

                # Compile to PDF
                with st.spinner("Converting to PDF..."):
                    pdf_content = compile_latex_to_pdf(cleaned_latex)

                if pdf_content:
                    st.session_state.pdf_content = pdf_content
                    st.success("PDF generated successfully!")
                    # Force a rerun to update the UI
                    st.rerun()

    with col2:
        st.header("Document Preview & Refinement")

        # Display PDF if available
        if st.session_state.pdf_content:
            # Display PDF
            display_pdf(st.session_state.pdf_content)

            # Provide download button with a unique key
            st.download_button(
                label="Download PDF",
                data=st.session_state.pdf_content,
                file_name="generated_document.pdf",
                mime="application/pdf",
                key="download_pdf_button"
            )

            # Display LaTeX source with toggle
            with st.expander("View LaTeX Source"):
                st.code(st.session_state.latex_content, language="latex")

            # Chat interface for document refinement
            st.subheader("Document Refinement Chat")
            st.markdown("Describe the changes you'd like to make to the document. The AI will update it accordingly.")

            # Display chat history
            chat_container = st.container()
            with chat_container:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['content']}")
                    else:
                        st.markdown(f"**AI:** {message['content']}")

            # Chat input for document refinement
            feedback = st.text_area("Describe your requested changes:", key="chat_input")

            if st.button("Update Document", key="update_document_button"):
                if not feedback:
                    st.warning("Please describe the changes you'd like to make.")
                else:
                    with st.spinner("Updating document..."):
                        updated_latex, summary = update_document(feedback)

                    if updated_latex:
                        # Update the stored LaTeX content
                        st.session_state.latex_content = updated_latex

                        # Store the update in chat history
                        st.session_state.chat_history.append({"role": "user", "content": feedback})
                        st.session_state.chat_history.append({"role": "assistant", "content": summary})

                        # Recompile to PDF
                        with st.spinner("Converting to PDF..."):
                            updated_pdf = compile_latex_to_pdf(updated_latex)

                        if updated_pdf:
                            st.session_state.pdf_content = updated_pdf
                            st.success("Document updated successfully!")

                            # Force a rerun to refresh the display
                            st.rerun()


if __name__ == "__main__":
    main()