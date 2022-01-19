test:
	cd src ; pipenv run pytest ../tests/

check_python_version:
	which python

# check python version before running, e.g., if pipenv was activated
