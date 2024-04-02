# Prompting for Rankings
_______________________________________________

All code can be executed as a stand-alone in the Jupyter notebook "tutorial.ipynb". The

## Set-up

```
git clone git@github.com:niklasstoehr/rankPrompt.git
cd /Users/username/Code/rankPrompt 
pip install -r requirements.txt
```

### [optional] Configure API keys in .env file

```
cd /Users/username/Code/rankPrompt_venv 
touch .env
vim .env
```
the content of the file should look as follows:

```
## Huggingface
HUGGINGFACE_KEY=<key>

## OpenAI 
OPEN_API_KEY=<key>
```

It can be loaded within the code as follows:

```
import dotenv
from transformers import AutoModelForCausalLM
dotenv.load_dotenv()

model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, token=os.getenv("HUGGINGFACE_KEY")).to(device)
```