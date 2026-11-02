.PHONY: up down migrate run worker check

up:
	docker compose up -d

down:
	docker compose down

migrate:
	python manage.py migrate

run:
	python manage.py runserver

worker:
	celery -A config worker -l info

check:
	python manage.py check
