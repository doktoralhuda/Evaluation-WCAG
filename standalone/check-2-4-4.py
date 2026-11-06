import sys
import warnings
import requests
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.agents import initialize_agent, AgentType
from langchain_anthropic import ChatAnthropic

warnings.filterwarnings("ignore")
load_dotenv()

if len(sys.argv) < 3:
    print('Penggunaan: python check-2-4-4.py "<url>" "<teks_tautan>"')
    sys.exit(1)

url = sys.argv[1]
link_text = sys.argv[2]


def browse(target_url: str) -> str:
    try:
        resp = requests.get(target_url, timeout=10)
        return resp.text[:3000]
    except Exception:
        return f"Webpage: {target_url}"


tool = Tool(
    name="browse_webpage",
    func=browse,
    description="Browse to a webpage URL and retrieve its content."
)

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

prompt = f"""Browse to {url} and summarize the content in one sentence. Then, tell me on a scale from 0 to 1 how suitable it would be to add a link to that website using this text for the anchor: '{link_text}'. Write your answer as a JSON object with the keys 'summary' and 'suitability'. For example {{"summary": "summary of the page", "suitability": a float number between 0 and 1}}"""

raw_output = agent.run(prompt)
data = JsonOutputParser().parse(raw_output)
suitability = float(data.get("suitability", 0.0))

print(f"URL             : {url}")
print(f"Teks Tautan     : {link_text}")
print(f"Skor Kesesuaian : {suitability}")
print(f"Status          : {'PASSED' if suitability >= 0.50 else 'FAILED'}")
