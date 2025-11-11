# NOTICE

This repository combines the Symbioza-DayMind MIT-licensed codebase with several third-party components. The dependencies listed below retain their original licenses. GPL tools run as standalone services and are not distributed as infringing binaries.

- **FastAPI** – MIT License  
- **Fava** – MIT License (runs as an external dashboard on port 5000; communicates via HTTP and shared ledger files)
- **Beancount** – GPLv2 (used externally for ledger exports and reporting; not linked into DayMind binaries)
- **Redis / redis-py** – BSD License
- **Whisper / OpenAI API** – Proprietary (used under OpenAI API terms)
- **Kivy** – MIT License
- **Terraform** – Mozilla Public License 2.0
- **PyDub** – MIT License
- **webrtcvad** – MIT License

The MIT-licensed DayMind runtime orchestrates these services via Text-First Storage (JSONL/text files) and HTTP while maintaining compliance with all listed licenses. For questions about licensing, contact the release manager before publishing the packaged artifact.
