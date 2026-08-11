# Streamlit UI

This directory contains the user-facing Streamlit application.

## Role of the UI

The UI is intentionally thin.

It should NOT:

- connect directly to MySQL;
- run SQL itself;
- call Groq directly;
- duplicate schema retrieval;
- duplicate LangGraph logic;
- implement authentication independently.

Instead:

```text
Streamlit
    |
    | HTTP + JWT
    v
FastAPI
    |
    v
Text-to-SQL agent
    |
    v
MySQL
```

## Planned user flow

1. User opens the Streamlit application.
2. User enters username/password.
3. UI calls `POST /auth/login`.
4. UI stores the returned bearer token in the current Streamlit session.
5. UI optionally calls `GET /me` to display the authenticated user/role.
6. User enters a natural-language question.
7. UI sends the question to `POST /query` with the bearer token.
8. UI displays:
   - generated SQL;
   - returned data;
   - row count;
   - attempt count;
   - errors when the agent fails.

## Backend dependency

Default local FastAPI address:

```text
http://127.0.0.1:8000
```

The UI should make this configurable rather than hard-coding production deployment assumptions.

## Current status

**Not implemented yet.**

Streamlit is installed and the `ui/` directory exists. The next implementation step is the actual Streamlit application.

## Design constraint

Keep the UI simple initially. The goal of this phase is to prove the complete authenticated user experience through the existing FastAPI contract.

Evaluation, advanced visualization, email, Excel export, and other tooling should be added after the core UI/API flow is stable.
