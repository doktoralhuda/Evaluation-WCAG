import sys
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv()

if len(sys.argv) < 3:
    print('Penggunaan: python similarity.py "<deskripsi_gambar>" "<teks_alternatif>"')
    sys.exit(1)

true_sentence = sys.argv[1]
candidate_sentence = sys.argv[2]

template = """Given the following true sentence:
 
    {true_sentence}

Tell me how similar is that sentence to the following one:

	{sentence}


Write your answer in a JSON object with one key: similarity (a float from 0 to 1) 
"""

llm = ChatOpenAI(model="gpt-5.6-terra")
prompt = PromptTemplate(template=template, input_variables=['true_sentence', 'sentence'])
parser = JsonOutputParser()

chain = prompt | llm | parser
data = chain.invoke({'true_sentence': true_sentence, 'sentence': candidate_sentence})
score = float(data.get("similarity", 0.0))

print(f"Deskripsi Gambar : {true_sentence}")
print(f"Teks Alternatif  : {candidate_sentence}")
print(f"Skor Similarity  : {score}")
print(f"Status           : {'PASSED' if score >= 0.50 else 'FAILED'}")
