# Sortformer two-speaker fixture sources

This directory contains two original FLAC utterances served by the
Hugging Face mirror of the LibriSpeech ASR corpus. The two rows have distinct
`speaker_id` values.

| File | LibriSpeech split | Speaker and reader | Chapter | Duration | SHA-256 |
|------|-------------------|--------------------|---------|----------|---------|
| `6930-75918-0000.flac` | `test-clean` | `6930`, Nolan Fout | `75918` | 3.505 s | `9ce35224156f071ab58eb7feb8a5ceae600f6f9f353da2a6cbf797b6b1ac8a23` |
| `7902-96591-0000.flac` | `test-other` | `7902`, Kyle Van DeGlast | `96591` | 2.095 s | `955d0cc57211f03986124307ade29fbab521e26133caa2d2831b9a1b38c7a9b6` |

Transcripts:

- `6930-75918-0000`: `CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS`
- `7902-96591-0000`: `I AM FROM THE CUTTER LYING OFF THE COAST`

## Source and attribution

LibriSpeech was prepared by Vassil Panayotov with the assistance of Daniel
Povey from LibriVox public-domain audiobooks. The corpus paper is by Vassil
Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur.

`LibriSpeech (c) 2014 by Vassil Panayotov`

- Canonical corpus page: <https://www.openslr.org/12>
- Hugging Face mirror: <https://huggingface.co/datasets/openslr/librispeech_asr>
- Mirrored dataset revision used here: `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`
- Corpus paper: <https://www.danielpovey.com/files/2015_icassp_librispeech.pdf>

The source archives and checksums published by OpenSLR are:

- [`test-clean.tar.gz`](https://www.openslr.org/resources/12/test-clean.tar.gz):
  MD5 `32fa31d27d2e1cad72775fee3f4849a9`
- [`test-other.tar.gz`](https://www.openslr.org/resources/12/test-other.tar.gz):
  MD5 `fb5a50374b501bb3bac4815ee91d3135`

The stored files match the corresponding official archive members byte for
byte. The adjacent `LICENSE.txt` is copied from the archives.

OpenSLR distributes LibriSpeech under the Creative Commons Attribution 4.0
International license: <https://creativecommons.org/licenses/by/4.0/>

The two stored FLAC files are the unmodified bytes returned by the Hugging Face
dataset server for row zero of `clean/test` and `other/test` at the revision
above. The integration test creates a derived 13-second signal only in memory.
It adds silence, repeats the first utterance, and linearly mixes the two sources
at half gain to create an overlap region. No endorsement by the corpus authors,
OpenSLR, LibriVox, or Hugging Face is implied.

The real-model test was validated with
`nvidia/diar_streaming_sortformer_4spk-v2` revision
`5240a64075176943f677d30fa2171c780229f341`. The `.nemo` file has SHA-256
`b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329`.
The test uses that cached revision automatically, or an explicit local model
set through `WLK_SORTFORMER_MODEL_PATH`.
