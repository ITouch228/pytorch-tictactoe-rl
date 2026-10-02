VENV = .venv

ifeq ($(OS),Windows_NT)
	BIN = $(VENV)/Scripts
else
	BIN = $(VENV)/bin
endif

PYTHON = $(BIN)/python
RUFF = $(BIN)/ruff

.DEFAULT_GOAL := help

.PHONY: help
help: ## Показать список команд
	@echo Доступные команды:
	@echo   install    Установить зависимости
	@echo   lint       Проверить код линтером ruff check
	@echo   format     Форматировать код ruff format
	@echo   fix        Исправить авто-ошибки и отформатировать
	@echo   test       Запустить тесты pytest
	@echo   train      Запустить обучение
	@echo   play       Играть против обученной сети
	@echo   minimax    Сыграть против минимакса
	@echo   check      Линтер + тесты
	@echo   clean      Очистить временные файлы

.PHONY: install
install: ## Установить зависимости
	$(PYTHON) -m pip install -r requirements.txt

# Линтинг и форматирование
.PHONY: lint
lint: ## Проверить код линтером (ruff check)
	$(RUFF) check .

.PHONY: format
format: ## Форматировать код (ruff format)
	$(RUFF) format .

.PHONY: fix
fix: ## Исправить авто-ошибки и отформатировать (ruff check --fix + ruff format)
	$(RUFF) check . --fix
	$(RUFF) format .

# Запуск обучения
.PHONY: train
train: ## Запустить обучение
	$(PYTHON) main.py

# Запуск игры
.PHONY: play
play: ## Играть против обученной сети
	$(PYTHON) play_vs_player.py

.PHONY: minimax
minimax: ## Сыграть против минимакса
	$(PYTHON) play_vs_minimax.py

.PHONY: clean
clean: ## Очистить временные файлы
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache') if p] "