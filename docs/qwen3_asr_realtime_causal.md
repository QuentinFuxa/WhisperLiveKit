# Qwen3-ASR Realtime Causal

Ce document explique la direction expérimentale pour transformer Qwen3-ASR,
qui est aujourd'hui pensé comme un modèle offline, en modèle ASR realtime
incrémental.

Le problème de départ est simple : dans le chemin Qwen3-ASR actuel, si on a
déjà traité `0:20` d'audio et qu'on reçoit une seconde de plus, le chemin naïf
ré-encode `0:21` complet. Le but du chantier realtime est de ne plus refaire
ce travail. On veut garder autant que possible les poids Qwen pré-entraînés,
mais exécuter l'audio tower avec un cache et un contexte borné.

## Résumé

On ne cherche pas seulement à accélérer le decoder texte. Le vrai coût vient du
fait que l'audio embedding est recalculé sur toute la fenêtre.

La cible realtime :

- encoder uniquement les nouvelles frames audio, plus un petit bord de contexte;
- conserver un cache KV audio entre deux chunks;
- conserver le cache KV du decoder texte;
- émettre des deltas de tokens au fil de l'eau;
- partir de l'audio encoder Qwen3-ASR au lieu de réentraîner un encodeur audio
  from scratch.

Speculative decoding et TurboQuant restent hors scope ici. Ils peuvent aider
ailleurs, mais ils ne corrigent pas le recalcul complet de l'audio tower.

## Chemin Qwen3-ASR Offline

Qwen3-ASR est conçu autour d'un prompt offline :

```text
audio complet
  |
  v
log-mel features
  |
  v
audio encoder Qwen sur toute l'utterance
  |
  v
embeddings audio injectés dans le prompt
  |
  v
decoder Qwen génère la transcription
```

Dans un wrapper streaming classique, on obtient souvent ça :

```text
update t = 20s:
  encode audio[0:20]
  decode transcript

update t = 21s:
  encode audio[0:21]
  decode transcript
```

Donc on paie encore une fois tout `audio[0:20]`. C'est exactement ce qu'on veut
supprimer.

## Chemin Realtime Cible

Le contrat change : le modèle ne reçoit plus "tout l'audio courant" à chaque
fois, il reçoit seulement le nouveau chunk et l'état précédent.

```text
nouveau chunk audio
  |
  v
append log-mel frames
  |
  v
audio encoder Qwen causalise
  |      garde un cache KV audio
  v
frames audio finalisées
  |
  v
projector / adapter vers dimension decoder
  |
  v
decoder step par groupe de frames
  |      garde un cache KV decoder
  v
[P] wait, [W] boundary, ou token texte
```

Le modèle ne répond plus à la question :

```text
"Quel est le transcript complet de la fenêtre actuelle ?"
```

Il répond plutôt à :

```text
"Vu l'état précédent et ces nouvelles frames finalisées, dois-je émettre
quelque chose maintenant ?"
```

## Pourquoi Pas Strictement Causal Dès Le Début ?

Un encodeur strictement causal impose :

```text
frame t peut regarder uniquement les frames <= t
```

C'est idéal pour la latence, mais c'est un gros choc de distribution pour un
modèle offline pré-entraîné avec du contexte futur.

La v2 utilise donc un compromis plus réaliste :

```text
left_context  = historique long, par exemple 15s
right_context = petit futur fixe, par exemple 640ms
```

Une frame n'est finalisée que lorsque son `right_context` est disponible. On
ajoute donc une latence fixe, mais on garde l'incrémentalité.

```text
temps ------------------------------------------------------------>

déjà finalisé              pending tail           nouveau chunk
====================       ............           +++++++++++++
cache KV réutilisé         attend son             encodé maintenant
                            right_context

après réception du nouveau chunk:

finalisé, y compris ancien pending tail             nouveau pending tail
======================================              ...................
```

Donc ce n'est pas forcément "zéro lookahead". C'est plutôt :

```text
streaming incrémental avec lookahead borné
```

C'est suffisant pour éviter de ré-encoder tout le passé.

## Chirurgie De L'Audio Encoder Qwen

L'idée importante : ne pas jeter l'audio tower Qwen. On garde les poids utiles,
et on change la manière de les exécuter.

On garde autant que possible :

- les poids du conv stem / input projection audio;
- les projections d'attention `q`, `k`, `v`, `o`;
- les MLP;
- les norms;
- le positional handling compatible;
- le decoder Qwen;
- le tokenizer;
- la LM head.

On change :

- le masque d'attention, qui devient local/streaming;
- l'exécution attention, qui utilise un cache KV append-only;
- l'éviction des vieux KV au-delà de `left_context`;
- le traitement du `right_context`, avec un `pending tail`;
- l'adapter/projector entre sorties audio et steps decoder;
- la stratégie de decoding, qui devient frame-synchrone.

Schéma attention :

```text
Attention offline originale:

frame i peut regarder:
  [0 ........................................ T]
   passé, présent, futur lointain


Attention streaming localisée:

frame i peut regarder:
              [i - left_context ...... i + right_context]
               passé cache             petit futur fixe
```

Les poids restent réutilisables parce que les projections `Q/K/V` apprennent
toujours le même type de représentation. Ce qui change, c'est l'ensemble des
clés/valeurs visibles à chaque frame et le fait qu'on les stocke entre chunks.

## Cache KV Audio

Pour chaque couche transformer audio, on conserve :

```text
K_cache: (batch, kv_heads, cached_seq, head_dim)
V_cache: (batch, kv_heads, cached_seq, head_dim)
```

À l'arrivée d'un nouveau chunk :

1. on calcule `Q/K/V` seulement pour les nouvelles frames et le bord nécessaire;
2. on append les nouveaux `K/V` au cache;
3. on applique l'attention locale sur les frames devenues finalisables;
4. on émet uniquement les sorties audio finalisées;
5. on garde les dernières frames en `pending tail`;
6. on évince les entrées plus anciennes que `left_context`.

Invariant attendu :

```text
travail par update ~= nouveau chunk + bord right_context
mémoire par session ~= left_context + right_context
```

Le coût ne doit plus grandir avec la durée totale de l'appel.

## Pending Tail

Le `pending tail` est ce qui permet d'utiliser un petit futur sans casser le
streaming.

Exemple avec `right_context = 640ms` :

```text
audio reçu:
  0s ----------------------------- 20.0s

safe à finaliser:
  0s ---------------------- 19.36s

pending:
                         19.36s --- 20.0s
```

Après réception d'une seconde en plus :

```text
audio reçu:
  0s ----------------------------------- 21.0s

safe à finaliser:
  0s ---------------------------- 20.36s

pending:
                               20.36s --- 21.0s
```

La région déjà finalisée n'est pas recalculée. Elle vit dans les caches.

## Decoder Frame-Synchrone

Dans le modèle offline, le decoder reçoit un bloc d'embeddings audio puis génère
du texte.

Dans le modèle realtime, on veut un contrat plus proche de :

```text
audio step n + état texte précédent -> prochaine décision token
```

On ajoute des tokens de contrôle :

- `[P]` : wait, ne rien émettre;
- `[W]` : boundary / stabilité mot;
- tokens Qwen normaux : texte réel.

Sans `[P]`, le modèle est poussé à parler trop souvent. C'est une des causes
des répétitions observées dans les premières versions : le système produit du
texte alors qu'il devrait attendre.

## Pourquoi Un Simple LoRA Ne Suffit Pas

LoRA peut aider à adapter les poids, mais il ne change pas tout seul le contrat
d'exécution.

Si le graphe fait encore :

```text
audio[0:t+1] -> encoder complet -> decoder
```

alors LoRA n'empêche pas de ré-encoder `audio[0:t]`.

Il faut d'abord rendre l'audio tower incrémentale :

```text
audio[t:t+1] -> encoder cache -> nouvelles frames finalisées
```

Ensuite seulement, LoRA/adapters deviennent utiles pour adapter le modèle au
nouveau masque local et au decoding frame-synchrone.

## Stages De Training

### Stage 0: Parité Du Wiring

Avant de faire du training lourd, il faut prouver que l'implémentation cache est
correcte.

Propriété attendue :

```text
sortie encoder chunked avec cache == sortie offline avec le même masque local
```

Ça ne veut pas dire qu'on matche le Qwen offline global original. Ça veut dire
que l'exécution streaming est mathématiquement cohérente.

### Stage A: Adapter D'abord

On freeze le maximum :

- decoder Qwen frozen;
- majorité de l'audio tower frozen;
- training du projector, gate, petits adapters.

Objectif : apprendre l'interface frame-synchrone sans détruire les
représentations Qwen.

### Stage A+: Anti-Répétition / Anti-Sur-Émission

Les premiers runs peuvent réduire la loss tout en devenant mauvais sur WLK :
sorties longues, répétitions `the/de`, texte émis trop souvent.

Stage A+ ajoute :

- métriques unigram/bigram/trigram repetition;
- ratio tokens texte émis;
- ratio wait/text;
- longueur d'hypothèse;
- decoding optionnel avec repetition penalty;
- no-repeat ngram optionnel;
- limite de tokens texte consécutifs;
- emit-rate loss;
- label smoothing sur frames texte.

Le but n'est pas de maquiller le modèle avec du decoding. Le but est de mesurer
le problème et de relancer des resumes courts régularisés.

### Stage B: LoRA Sur Audio Tower

Si Stage A est correct mais manque de qualité, on ajoute du LoRA léger sur :

- projections attention audio;
- MLP audio;
- probablement d'abord les dernières couches audio.

Objectif : laisser l'encoder Qwen s'adapter au contexte local/right-context
sans fine-tuner toute la pile.

### Stage C: Distillation

On peut utiliser Qwen3-ASR offline, surtout `1.7B`, comme teacher :

- teacher produit transcripts ou pseudo-labels de meilleure qualité;
- aligner force les mots/subwords sur la timeline audio;
- student realtime apprend sous contrainte streaming.

## Alignement

Le realtime training a besoin de cibles frame-synchrones, pas seulement d'un
transcript complet.

Sources possibles :

- timestamps natifs des datasets;
- forced alignment;
- Qwen3 ForcedAligner si exploitable;
- WhisperX / MFA;
- pseudo-transcripts teacher si les labels sont faibles.

On peut échantillonner un délai cible `tau`, par exemple de `240ms` à `2400ms`,
pour apprendre plusieurs compromis latence/qualité.

## Contrat Serving

Une session realtime doit être stateful :

```text
session_state = {
  conv_history,
  audio_kv_cache,
  pending_audio_tail,
  decoder_kv_cache,
  emitted_tokens,
  anti_repetition_state
}
```

Chaque appel `append_audio` :

```text
1. reçoit seulement les nouvelles frames PCM/log-mel
2. met à jour les caches conv/audio
3. finalise les frames dont le right_context est disponible
4. exécute les steps decoder correspondants
5. émet les tokens non-[P]
6. garde les caches pour l'appel suivant
```

Différence majeure avec un endpoint STT classique :

```text
STT classique:
  fichier audio complet -> transcript final

Realtime natif:
  nouveau chunk + état session -> delta texte + nouvel état session
```

## Critères De Succès

Pour un stream où le client a déjà envoyé `0:20` puis ajoute `+1s`, on veut :

- aucun ré-embedding complet de `0:20`;
- compute audio proportionnel à environ `1s + right_context`;
- mémoire bornée par fenêtre de cache;
- texte stable avec peu de révisions;
- qualité assez proche du Qwen offline pour justifier ensuite un port Metal.

Le premier gate est technique :

```text
chunked cached audio == référence offline masquée
```

Ensuite seulement on dépense du H100 sur les longs trainings et les benchmarks
WER/WLK.

## Migration En Une Image

Avant :

```text
audio[0:t]   -> full audio encoder -> full prompt -> decoder -> transcript
audio[0:t+1] -> full audio encoder -> full prompt -> decoder -> transcript
```

Après :

```text
audio[t:t+1] -> cached local audio encoder -> frames finalisées
             -> cached decoder steps       -> deltas texte
```

C'est pour ça que le chantier est architectural. Le training est nécessaire,
mais il vient après le changement de contrat : passer d'un encodeur offline
global à un encodeur audio incrémental avec cache et contexte borné.
