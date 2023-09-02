# jarvis
test equipment diagnostic expert

pdf2db uses the sub-folder name as manufacturers or make and file name as model number
then the information is extracted from the database based on relevant model number keywords and circuit troubleshooting principles.

set up virtual env
`python -m  venv venv

enter env
source venv/bin/activate

exit = `deactivate`

install deps with 
`pip install -r requirements.txt`

(You will need to install https://github.com/tesseract-ocr/tesseract,
https://github.com/RedisAI/RedisAI (use the docker version) if you plan on using pdf2db script)

you can run it with 
`python -m jarvis`
