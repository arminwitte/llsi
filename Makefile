update:
	python3 -m pip install --upgrade pip
	python3 -m pip install --upgrade isort black coverage pytest build twine
    
isort:
	isort src/ tests/
    
black: isort
	black src/ tests/

test:
	coverage run -m pytest
	
coverage: test
	coverage report

build:
	python3 -m build
	
upload:
	python3 -m twine --skip-existing dist/*
