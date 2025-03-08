
to use

uv venv .venv
.venv\Scripts\activate

#make sure to set interpreter in your code editor so intellisense works

uv pip install -r requirements.txt
uv run python main.py

#remove venv when done using
deactivate