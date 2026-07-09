"""Per-session translation helpers.

The server loads a single shared NLLB model (see ``TranscriptionEngine``).
NLLB selects the output language per generation call: the transformers
backend passes ``forced_bos_token_id`` and the ctranslate2 backend a
``target_prefix``, so one loaded model can serve several target languages
concurrently. Each session already owns its own ``OnlineTranslation``
instance (with its own buffers and target token), so a per-session target
only requires building that instance with a different target language.
No lock or model reload is needed.
"""

import logging

logger = logging.getLogger(__name__)


def session_translation_factory(args, translation_model, target_language):
    """Create an OnlineTranslation for a session-specific target language.

    Shares the server-wide translation model; only the per-session decoder
    target differs. Falls back to the server-wide target (with a warning)
    if the requested language is not recognized.
    """
    from nllw import OnlineTranslation

    from whisperlivekit.core import _nllw_language_code, online_translation_factory

    source_language = _nllw_language_code(args.lan)
    session_target = _nllw_language_code(target_language)
    try:
        return OnlineTranslation(translation_model, [source_language], [session_target])
    except ValueError as e:
        logger.warning(
            "Per-session target_language=%r rejected (%s); "
            "falling back to server-wide target %r.",
            target_language, e, args.target_language,
        )
        return online_translation_factory(args, translation_model)
