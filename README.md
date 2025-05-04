# CrewAI Assistant

A voice-enabled AI assistant that combines the power of OpenAI's API with CrewAI for advanced capabilities.

## Features

- Voice chat with natural language processing
- Fast responses using direct OpenAI API
- Advanced capabilities using CrewAI when needed
- Memory system to remember past conversations
- Bilingual support (Finnish and English)

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/CrewAI_Assistant.git
cd CrewAI_Assistant
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the assistant from the project root:
```
python run.py
```

All persistent data (memories, ideas, conversation history, tasks) is stored in the `knowledge/` directory at the project root.

Make sure your `.env` file is in the project root and contains your OpenAI API key (and optionally Google Cloud credentials):
```
OPENAI_API_KEY=your_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=absolute_path_to_your_google_credentials.json
```

### Voice Commands

- Say "käytä agenttia" to switch to CrewAI mode for advanced capabilities
- Say "käytä puhetta" to switch back to direct OpenAI mode for faster responses

## License

MIT 