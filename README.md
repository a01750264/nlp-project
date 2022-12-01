# nlp-project
### To run this project you need to use Python 3.9, this is to avoid errors with Flair library which is not available in higher versions of Python.

This project uses a Deepl API key, see the [docs](https://www.deepl.com/docs-api/translate-text/translate-text/) to create it, then create a .env file which contains the following:
- DEEPL_API_KEY='your-deepl-api-key'

#### Instructions for running the code:
- First, in the project's directory, create a virtual environment in which you specify python version 3.9: ```virtualenv -p 'Path/to/your/Python3.9/executable' venv```
- Next, activate the virtual environment in the project: ```venv/bin/activate``` or ```venv\Scripts\activate```
- Then run the following command to install the required libraries and to run the code: ```pip install -r requirements.txt && python run.py```

## Second exercise loss graph
![Training loss graph](https://github.com/a01750264/nlp-project/blob/main/loss.JPG?raw=true)
