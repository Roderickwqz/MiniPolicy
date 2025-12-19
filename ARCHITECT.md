mini_policy/
  app/
    main.py                 # FastAPI
    api/                    # routes
      runs.py
      events.py
    core/
      config.py
      logging.py
      schemas.py            # Pydantic contracts (skills/events/artifacts)
      evidence.py           # evidence contract helpers
    orchestration/
      graph.py              # LangGraph definition (先用 stub graph)
      tasks.py              # Celery task entrypoints
      state.py              # run state machine + checkpoints pointer
    tools_mcp/
      gateway.py            # Tool Permission + Guardrails Gateway (先最简 allowlist)
      pdf_ingest.py         # MCP-style wrapper (先本地实现，接口先对齐)
      vector_index.py
      graph_store.py        # Neo4j later; P0可空实现
    storage/
      db.py                 # SQLite/Postgres (推荐先 SQLite)
      models.py             # Run, Event, Artifact
      repo.py               # persistence
    reporting/
      report_md.py
      report_pdf.py
  worker/
    celery_app.py
  docker-compose.yml        # redis + (optional) neo4j later
  README.md
