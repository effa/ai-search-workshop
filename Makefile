run:
	source env/bin/activate
	jupyter notebook

install:
	python3 -m venv env/
	source env/bin/activate
	pip3 install -r requirements.txt
