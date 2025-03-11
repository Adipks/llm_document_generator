# llm_document_generator
uses an llm to automatically generate any kind of Document based on the context provided

This repository provides a streamlined solution for generating  documents using `Streamlit`, `OpenAI`'s LLM's

##  Getting Started
Follow these steps to set up and run the project on your system.

### Step 1: Clone the Repository
Run the following command to clone the repository to your local machine:

```bash
git clone https://github.com/Adipks/llm_document_generator.git
cd llm_document_generator
```

### Step 2: Create a Virtual Environment
It's recommended to create a virtual environment to manage dependencies:

```bash
python3 -m venv doc_env
source doc_env/bin/activate  # On Linux/MacOS
```

On Windows:
```bash
doc_env\Scripts\activate
```

### Step 3: Install System Dependencies
Install the required system packages for document generation:

```bash
sudo apt-get install texlive-full pandoc
```

### Step 4: Install Python Dependencies
Install the necessary Python packages using `pip`:

```bash
pip install streamlit openai python-dotenv python-docx
```

### Step 5: Run the Application
To launch the document generator, run the following command:

```bash
streamlit run docu.py
```

## 📋 Additional Notes
- Ensure your `.env` file is properly configured with your OpenAI API key for successful execution.
- For troubleshooting or issues, please refer to the official documentation of the respective dependencies.

## 💬 Need Help?
Feel free to raise an issue or contact me for support.

Happy Documenting! 😊
