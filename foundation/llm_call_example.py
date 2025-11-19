from dotenv import load_dotenv
import os
from openai import OpenAI

def load_envrionment():
    # read all environment values from the .env file
    load_dotenv(override=True)

def validate_environment() -> bool:
    # validate that API keys are set
    if(os.getenv("OPENAI_API_KEY")):
        print(f"OpenAI API key is set - starts with {os.getenv("OPENAI_API_KEY")[:8]}\n")
    else:
        print(f"OpenAI API key missing...\n")
        return False

    return True

def get_system_prompt():
    return """You are a helpful assistant"""

def get_user_prompt():
    return """What is the capital of France?"""

def get_message():
    return [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": get_user_prompt()}
    ]

def get_model():
    return "gpt-4o-mini"

def run_llm():
    openai_client = OpenAI()
    response = openai_client.chat.completions.create (
        messages=get_message(),
        model=get_model()
    )

    print(f"User Query: {get_user_prompt()}")
    print(f"LLM Response: {response.choices[0].message.content}")


def main():
    load_envrionment()
    valid = validate_environment()
    if(valid):
        run_llm()

if __name__ == "__main__":
    main()