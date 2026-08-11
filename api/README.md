# API Layer

This directory contains the FastAPI service layer for the Text-to-SQL Agent.

## Responsibilities

The API layer is responsible for:

1. Authenticating users with JWT.
2. Applying role-based table access.
3. Receiving natural-language questions.
4. Running schema retrieval and FK expansion.
5. Calling the existing agent/self-correction pipeline.
6. Returning SQL and query results as JSON.
7. Logging request identity, SQL, status, attempts, errors, and latency.

## Files

### `auth.py`

Provides:

- password hashing/verification with bcrypt;
- application user storage;
- JWT creation;
- JWT validation;
- `get_current_user` FastAPI dependency;
- `require_admin` dependency;
- initial `app_users` table setup.

The application user database is separate from the target data-access policy. The SQL agent continues to use the configured read-only database credential for executing generated queries.

### `table_access.py`

Provides the application-level role/table enforcement point.

Current roles:

- `standard`
- `admin`

The current policy is permissive, so both roles can access the current Sakila tables. Restrictions can later be added through `ROLE_TABLE_RESTRICTIONS`.

### `main.py`

Wires the API request pipeline:

```text
JWT authentication
    -> table access filtering
    -> schema retrieval
    -> FK expansion/ranking
    -> LangGraph agent
    -> JSON response
```

## Endpoints

### `POST /auth/login`

Accepts OAuth2 password-form credentials.

Returns:

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer"
}
```

### `GET /me`

Requires:

```text
Authorization: Bearer <token>
```

Returns the authenticated username and role.

### `POST /query`

Requires authentication.

Request:

```json
{
  "question": "Which customers rented the most movies?"
}
```

Successful response contains:

- generated SQL;
- result columns;
- result rows;
- row count;
- truncation flag;
- number of agent attempts.

### `GET /health`

Returns:

```json
{
  "status": "ok"
}
```

## Running

From the project root:

```powershell
uvicorn api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Current status

**Complete and tested.**

Authentication, `/me`, and `/query` have been successfully exercised through FastAPI.
