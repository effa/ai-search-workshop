run:
	source env/bin/activate
	jupyter notebook

install:
	python3 -m venv env/
	source env/bin/activate
	pip3 install -r requirements.txt
	jupyter nbextension enable --py widgetsnbextension --sys-prefix

install-conda:
	conda create -n ai-search python=3.6 anaconda
	source activate ai-search
	pip install robomission
	jupyter nbextension enable --py widgetsnbextension --sys-prefix

run-conda:
	source activate ai-search
	jupyter notebook
