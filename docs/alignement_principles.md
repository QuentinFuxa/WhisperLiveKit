### Alignment between STT Tokens and Diarization Segments 

- Example 1: The punctuation from STT and the speaker change from Diariation come in the prediction `t`
- Example 2: The punctuation from STT comes from prediction `t`, but the speaker change from Diariation come in the prediction `t-1`
- Example 3: The punctuation from STT comes from prediction `t-1`, but the speaker change from Diariation come in the prediction `t`

> `#` Is the split between the `t-1` prediction and `t` prediction.  


## Example 1:
```text
punctuations_segments : __#_______.__________________!____
diarization_segments:
SPK1                    __#____________
SPK2                      #            ___________________
-->
ALIGNED SPK1            __#_______.
ALIGNED SPK2              #        __________________!____

t-1 output:
SPK1:                   __#
SPK2: NO
DIARIZATION BUFFER: NO

t output:
SPK1:                       __#__.
SPK2:                             __________________!____
DIARIZATION BUFFER: No
```

## Example 2:
```text
punctuations_segments : _____#__.___________
diarization_segments:
SPK1                    ___  #
SPK2                       __#______________
-->
ALIGNED SPK1            _____#__.
ALIGNED SPK2                 #   ___________

t-1 output:
SPK1:                   ___  #
SPK2:
DIARIZATION BUFFER:        __#

t output:
SPK1:                      __#__.
SPK2:                            ___________
DIARIZATION BUFFER: No
```

## Example 3:
```text
punctuations_segments : ___.__#__________
diarization_segments:
SPK1                    ______#__
SPK2                          #  ________
-->
ALIGNED SPK1            ___.  #
ALIGNED SPK2                __#__________

t-1 output:
SPK1:                   ___.  #
SPK2:
DIARIZATION BUFFER:         __#

t output:
SPK1:                         #
SPK2:                       __#___________
DIARIZATION BUFFER: NO
```
