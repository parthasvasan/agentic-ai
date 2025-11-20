from dotenv import load_dotenv
import os
from openai import OpenAI
from anthropic import Anthropic
from ast import List
import json


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

    if(os.getenv("ANTHROPIC_API_KEY")):
        print(f"Anthropic API key is set - starts with {os.getenv("ANTHROPIC_API_KEY")[:7]}\n")
    else:
        print(f"Anthropic API key missing...\n")
        return False

    if(os.getenv("GROQ_API_KEY")):
        print(f"Groq API key is set - starts with {os.getenv("GROQ_API_KEY")[:4]}\n")
    else:
        print(f"Groq API key missing...\n")
        return False

    if(os.getenv("GOOGLE_API_KEY")):
        print(f"Google API key is set - starts with {os.getenv("GOOGLE_API_KEY")[:2]}\n")
    else:
        print(f"Google API key missing...\n")
        return False

    return True

def get_evaluator_system_prompt() -> str:
    return f"""You are an expert in cyber security. Your job is to review and rank reports prepared by
    other experts in the cyber security industry
    """

def get_evaluator_user_prompt(consolidated: str) -> str:
    return f"""Evaluate the following responses from different
    LLMs on the top concern for organzations in the area of container security in 2026. Return the rankings and the various factors you used part of your rubrik to come
    up with rankings. Respond with JSON, and only JSON, with the following format:
    {{"rubric": {{"factor1": "weight1%", "factor2": "weight2%", ...}}, "results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

    Also, include a rubric that you used to rank these competitors as a reference.

    Here are the responses from each competitor:

    {consolidated}

    Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""


def get_participant_system_prompt() -> str:
    return """You are a helpful cyber security consultant. You have expertise in the field of
    cyber security and your job is primarily to provide recommendation to organizations on developing
    robust strategies for securing their organization."""

def get_participant_user_prompt() -> str:
    return  """Prepare a recommendation to implemment a robust security strategy for containers built 
    and deployed across various business units. In your recommendation, describe what should be the 
    our top most concern in 2026 to address in the area of container security. Include clear, and 
    consise recommendation along with reasoning."""

def get_participant_message() -> List:
    return [
        {"role": "system", "content": get_participant_system_prompt()},
        {"role": "user", "content": get_participant_user_prompt()}
    ]

def get_evaluator_message(consolidated: str) -> List:
    return [
        {"role": "system", "content": get_evaluator_system_prompt()},
        {"role": "user", "content": get_evaluator_user_prompt(consolidated)}
    ]


def run_openai() -> str:
    openai_client = OpenAI()
    resp = openai_client.chat.completions.create(
        messages=get_participant_message(),
        model="gpt-4o-mini"
    )
    return resp.choices[0].message.content

def run_anthropic() -> str:
    anthropic_client = Anthropic()
    resp = anthropic_client.messages.create(
        messages=[{"role": "user", "content": get_participant_user_prompt()}],
        system=get_participant_system_prompt(),
        model="claude-sonnet-4-5",
        max_tokens=200
    )
    return resp.content[0].text

def run_groq() -> str:
    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
    resp = groq_client.chat.completions.create(
        messages=get_participant_message(),
        model="openai/gpt-oss-120b"
    )
    return resp.choices[0].message.content

def run_evaluator_llm(consolidated) -> str:
    google_client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
        api_key=os.getenv("GOOGLE_API_KEY")
        )
    resp = google_client.chat.completions.create(
        messages=get_evaluator_message(consolidated),
        model="gemini-2.5-flash"
    )
    return resp.choices[0].message.content

def run_llm():
    '''
    Ask 3 of the 4 LLMs to identify the top most concern for organization in a certain area and
    have them expalin the reasoning. Use the 4th LLM to evaluate their responses and rank them
    using rubrik. Have the evaluator output the rankings as well as the rubrik it used for evaluation.
    '''
    recommendations = []

    print(f"Calling the LLM# 1 - OpenAI...")
    response = run_openai()
    recommendations.append(response)

    print(f"Calling the LLM# 2 - Anthropic...")
    response = run_anthropic()
    recommendations.append(response)

    print(f"Calling the LLM# 3 - Groq...")
    response = run_groq()
    recommendations.append(response)

    # Combine all 3 responses into a single string to provide as input to the evaluator LLM
    consolidated = ""
    for index,resp in enumerate(recommendations):
        consolidated += f"#Reponse from participant {index+1}: \n\n{resp}\n\n"

    print(f"Calling the evaluator LLM - Gemini...")
    response = run_evaluator_llm(consolidated)
    print(response)
    
def main():
    load_envrionment()
    valid = validate_environment()
    if(valid):
        run_llm()

if __name__ == "__main__":
    main()
