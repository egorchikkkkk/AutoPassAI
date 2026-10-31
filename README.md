# AutoPassAI

Асинхронный сервис на FastAPI: проверка доступности приложения, версии установленного пакета и подключения к PostgreSQL. Проект включает Docker Compose, автоматические проверки и публикацию Docker-образов в GitHub Container Registry (GHCR).

## Требования

- **Контейнерный запуск:** Docker с Docker Compose.
- **Локальная разработка:** uv;
- **Оболочка:** команды приведены для Bash; на Windows используйте WSL.
- **Порты:** свободные `8000` для приложения и `5432` для PostgreSQL.

Все команды выполняются из корня репозитория.

## Быстрый запуск

```bash
docker compose up -d --build --wait
```

Compose собирает приложение и запускает его после готовности PostgreSQL.

- **API:** http://localhost:8000
- **Swagger:** http://localhost:8000/docs

```bash
curl http://localhost:8000/healthz
# {"status":"ok"}

docker compose ps             # состояния сервисов
docker compose logs -f app    # логи приложения;
docker compose down          # остановка и удаление контейнеров и сети
```

Данные PostgreSQL сохраняются в томе `postgres_data`. Команда `docker compose down -v` удаляет также том с данными.

После изменения исходников:

```bash
docker compose up -d --build --wait app
```

## Локальная разработка

PostgreSQL работает в контейнере, приложение — в локальном Python-окружении:

```bash
uv python install 3.13
uv sync --locked
cp -n .env.example .env
docker compose up -d --wait db
uv run uvicorn autopassai.main:app --host 127.0.0.1 --port 8000 --reload
```

`Ctrl+C` останавливает локальный сервер. Если сервис `app` уже запущен в Compose, сначала остановите его через `docker compose stop app`, чтобы освободить порт `8000`.

## Конфигурация

- `APP_NAME` — название приложения.
- `DATABASE_URL` — строка подключения SQLAlchemy к PostgreSQL через asyncpg.

При локальном запуске настройки загружаются из окружения и `.env`; переменные окружения имеют приоритет. Пример — [.env.example](.env.example).

```dotenv
APP_NAME=AutoPassAI
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/autopassai
```

В текущем [docker-compose.yml](docker-compose.yml) настройки заданы через `environment`: база `autopassai`, пользователь и пароль `postgres`. Приложение подключается к `db:5432`, а с компьютера БД доступна по `localhost:5432`. Изменение локальной `.env` не заменяет явно заданные значения Compose.

Параметры `POSTGRES_*` применяются при первой инициализации пустого тома, а не меняют существующего пользователя или базу.

## API

- `GET /healthz` — доступность приложения без проверки БД.
- `GET /api/v1/version` — версия установленного пакета AutoPassAI.
- `GET /api/v1/health` — состояние PostgreSQL, версия сервера и время ответа.

При ошибке БД `/api/v1/health` возвращает **HTTP 200** с `"status": "error"` в JSON. Прикладные таблицы не создаются: проверка выполняет `SELECT version()`.

Подключение к SQL-консоли:

```bash
docker compose exec db psql -U postgres -d autopassai
```

## Проверки

Однократно для каждого локального клона установите Git-хук:

```bash
uv run pre-commit install
```

При коммите выполняются Ruff, проверка форматирования, размера файлов и завершающих пробелов. Установка пакета pre-commit через uv сама по себе Git-хук не регистрирует.

Проверки изменений вручную:

```bash
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m pytest
uv run coverage report -m
```

Все хуки для отслеживаемых файлов:

```bash
uv run pre-commit run --all-files
```

Тесты используют моки БД и не требуют PostgreSQL. Локальные проверки дают обратную связь до отправки изменений; CI выполняет проверки независимо.

## CI/CD

- **CI** ([ci.yml](.github/workflows/ci.yml)) — push в `main` и pull request в `main`: Ruff, форматирование, тесты и coverage.
- **CD** ([cd.yml](.github/workflows/cd.yml)) — push Git-тега по шаблону `v*.*.*`: проверка версии, сборка и публикация образа.
- **Реестр:** `ghcr.io/egorchikkkkk/autopassai`; теги образа — версия с префиксом `v` и `latest`.

CD не ожидает успешного CI автоматически и не разворачивает приложение на сервере. Текущий Compose собирает образ локально, а не загружает опубликованный из GHCR.

### Выпуск версии

1. Измените `version` в `pyproject.toml` и ожидаемую версию в `tests/test_health.py`; обновите README при изменении возможностей или запуска.
2. Выполните `uv lock`, `uv sync --locked` и проверки.
3. Закоммитьте изменения, проведите PR в `main` и проверьте успешный CI.
4. Отметьте принятый коммит новым Git-тегом и отправьте его.

Например, для **ещё не выпущенной** версии `0.1.3`, после принятия соответствующих изменений:

```bash
git switch main
git pull --ff-only
git status --short
# Продолжайте при чистом рабочем дереве и проверенной версии 0.1.3.
git tag -a v0.1.3 -m "Release 0.1.3"
git push origin v0.1.3
```

Тег без начальной `v` должен совпадать с версией в `pyproject.toml`; иначе CD завершится ошибкой. Отправка только коммитов без тега публикацию не запускает.

## Структура проекта

```text
src/autopassai/
├── main.py          # ASGI-приложение и регистрация маршрутов
├── config.py        # конфигурация и версия пакета
├── logging.py       # настройка логирования
├── api/             # HTTP-маршруты
├── db/              # engine и фабрика сессий SQLAlchemy
└── services/        # проверка PostgreSQL
tests/               # тесты API и сервиса проверки БД
.github/workflows/   # CI и публикация образа
pyproject.toml       # метаданные, зависимости и настройки инструментов
uv.lock              # зафиксированные зависимости
Dockerfile           # сборка образа приложения
docker-compose.yml   # приложение, PostgreSQL, сеть и том
```
