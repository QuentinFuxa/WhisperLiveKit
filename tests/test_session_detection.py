from src.gpt_postproc.processor import _assign_sessions


def test_session_increment_logic() -> None:
    segments = [
        {"text": "Ahoj", "start": 0.0, "end": 1.0},
        {"text": "Přidej poznámku", "start": 5.0, "end": 6.0},
        {"text": "Vytvoř Anki kartičky", "start": 25.0, "end": 27.0},
    ]

    enriched = _assign_sessions(segments, gap_threshold=10.0)
    session_ids = [session_id for (_segment, session_id, _gap) in enriched]

    assert session_ids[0] == session_ids[1]
    assert session_ids[2] > session_ids[1]
