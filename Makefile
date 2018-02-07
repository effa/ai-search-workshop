# Using local install without isolation.
# For isolated environments, see the approaches below.
install:
	pip3 install -r requirements.txt --user
	~/.local/bin/jupyter notebook nbextension enable --py widgetsnbextension --sys-prefix

run:
	~/.local/bin/jupyter notebook


## ----------------------------------------------------------------------
## Using virtualenv and virutalenvwrapper
## Assumes virualenvwrapper is set:
## export WORKON_HOME=~/.virtualenv
## source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
#install:
#	mkvirtualenv -a . ai-search
#	workon ai-search
#	pip3 install -r requirements.txt
#	jupyter nbextension enable --py widgetsnbextension --sys-prefix
#
#run:
#	workon ai-search
#	jupyter notebook
#
#
## ----------------------------------------------------------------------
## Using Python3 venv
#
#install:
#	python3 -m venv env/
#	source env/bin/activate
#	pip3 install -r requirements.txt
#	jupyter nbextension enable --py widgetsnbextension --sys-prefix
#
#run:
#	source env/bin/activate
#	jupyter notebook
#
## ----------------------------------------------------------------------
## Using anaconda
#
#install-conda:
#	conda create -n ai-search python=3.6 anaconda  # TODO: + jupyter, matplotlib, pandas
#	source activate ai-search
#	jupyter nbextension enable --py widgetsnbextension --sys-prefix
#
#run-conda:
#	source activate ai-search
#	jupyter notebook
