# Roadmap — atteindre les perfs Voxtral sans palliatifs (2026-06-11)

Réponse à : « comment faire pour avoir les mêmes perfs que Voxtral sans palliatifs bancals ? »
Sources : papier Voxtral Realtime complet (arXiv:2602.11298, HTML v1), littérature
streaming-ASR (Narayanan ASRU'19 RSP, Kurata & Saon, CoDERT, Tang, Hori, U2/Emformer/
Zipformer/icefall/CUSIDE), confrontée à nos résultats (WS2, WS2c, D2a).

## Ce que fait réellement Voxtral Realtime

- **Encodeur 970M strictement causal entraîné FROM SCRATCH** (32 couches, RMSNorm/SwiGLU/RoPE,
  sortie 50 Hz, fenêtre glissante **15 s** = 750 frames, conv stem causal). Pas de blocs
  bidirectionnels, pas de lookahead encodeur.
- **Aucune distillation nulle part.** Supervision = CE next-token pure sur cibles
  frame-synchrones [P]/[W] construites depuis des timestamps mot. Deux phases :
  warm-up encodeur décodeur gelé (5 % du budget), puis joint end-to-end (95 %).
- Le délai vit dans le **décodeur** (boucle 1 token / 80 ms, [P] jusqu'à mot complet + tau
  écoulé) ; tau conditionné par **Ada RMS-Norm** (gating multiplicatif FFN, 5M params).
- **z-loss** pour empêcher les embeddings texte d'écraser l'audio (sinon le modèle ignore l'audio).
- Long-form : fenêtres glissantes des deux côtés, « infinite streaming »… **MAIS** 16–32 frames
  de **left-padding (attention sinks) à l'inférence** font passer le WER long-form de
  10.98 → 7.73. Même Voxtral, entraîné end-to-end à l'échelle industrielle, a besoin de ce
  « palliatif » à l'inférence.
- Données/compute non divulgués (« large-scale, 13 langues » ; batchs de 370 h audio →
  centaines de milliers d'heures probables).

Implications pour nous : leur fenêtre encodeur (15 s) = la nôtre — la fenêtre n'est pas notre
problème. Ce qui nous manque : objectif task-level sur longues séquences + exposition à l'état
porté (state exposure) + sinks. Notre distillation décodeur-gelé ≈ seulement leur phase 5 %.

## Verdict scientifique (hypothèse demi-validée)

- **VALIDÉ** : l'imitation d'embeddings frame-à-frame sur longues chaînes est structurellement
  condamnée, pas limitée par le budget. La cible frame-t du teacher offline encode jusqu'à 64 s
  de contexte bidirectionnel qu'un étudiant causal 15 s-left **ne peut pas posséder**
  (Kurata & Saon : la distillation offline→streaming naïve est destructive, 23.3→33.0 ;
  CoDERT : feature-matching teacher gelé « nearly futile » ; Tang : MSE caché seul PIRE que
  pas de distillation). Le plancher de loss ×2-3 de WS2c = cette littérature reproduite.
- **NON VALIDÉ** : la task-loss seule ne corrige PAS le drift. Narayanan ASRU'19 : un modèle
  streaming entraîné en pure task-loss s'effondre quand même en session longue (4.9 → 49 % WER,
  dominé par les délétions — exactement notre signature « début propre puis bouillie »).
  Le correctif opérant = **Random State Passing** (initialiser l'état d'entraînement depuis des
  états portés réalistes) : 49.0 → 16.2, sans gradients cross-batch, coût ≈ séquence courte.
- La task-loss rend le drift **corrigeable** (ses cibles sont atteignables sous contrainte
  causale, contrairement aux embeddings offline) ; le **state-exposure** est ce qui le corrige.
- Le reset-on-rollover à 0.187 n'est PAS un hack honteux : Emformer, icefall et WeNet bornent
  l'état en permanence par construction. C'est l'état de l'art de la pratique.

## Plan par phases

### Phase 0 — diagnostics, ~20 $, une soirée H100 (DÉCIDE TOUT LE RESTE)
1. **Sinks à l'inférence sur notre chaîne causale** : épingler 16–32 frames (zéro-audio ou
   premières frames jamais éjectées) en tête du KV causal, éval 3 fichiers longs SANS reset.
   Plus gros levier long-form de Voxtral (−3.25 WER). Mécanisme plausible de notre drift :
   quand la fenêtre 15 s glisse, les premières clés (puits d'attention naturels) sont éjectées
   → StreamingLLM. Le reset marche peut-être simplement parce qu'il recrée un « début ».
   Si les sinks récupèrent l'essentiel de 0.95→0.26 : ship, retirer le reset à coût ~nul.
2. **Courbe CE vs profondeur de chaîne** : décodeur gelé teacher-forcé sur les embeddings d'une
   chaîne de 300 blocs, CE par bloc. Montée corrélée à la profondeur = il y a du gradient
   task-level exactement là où vit le drift (prémisse de WS3). Courbe plate = prémisse falsifiée
   → on s'arrête là, on garde reset + backend Voxtral.
3. **Probe carry-in d'état** : initialiser le cache encodeur d'un segment frais depuis un cache
   mi-fichier sauvegardé ; dégradation immédiate = mismatch de distribution d'état confirmé
   (précondition de RSP).
4. **Barre honnête** : backend Voxtral Realtime existant (voxtral_hf_streaming) sur 5 fichiers
   MCIF à tau=480/960 ms + 32 sinks, scoring human-ref whisper-norm, RTF + VRAM pic.

### Phase 1 — WS3, state-exposure + objectif task-level, 75–165 $, ~55 % de succès
Warm-start `runs/jl_ws2_mix_pos_p2b/tower_last.pt`. Objectif : **CE teacher-forcé à travers le
décodeur Qwen3 GELÉ** (encodeur+adapter entraînables), option KL sur logits vs le même décodeur
nourri d'embeddings offline (in-place distillation à la Dual-mode ASR — safe car les deux
branches scorent les mêmes positions texte). MSE frame reléguée à auxiliaire ≤0.01.
Le vrai fix, state exposure :
- **RSP** : p=0.5, init du cache KV encodeur depuis des caches finaux sauvegardés (batchs
  précédents ou rollouts longs réels) ;
- concaténations 16–96 s avec **loss sur la dernière utterance seulement** (WS2c gaspillait le
  gradient sur les positions à état peu profond) ;
- randomisation de la profondeur de chaîne portée {2, 8, 16, 64, ∞} + augmentations
  offset/taille de bloc existantes ;
- **16–32 sink frames épinglées en permanence** dans le KV causal ;
- 50 % de batchs courts cold-start (mixing U2) pour protéger les gates courts 0.21/0.20 ;
- triple-gate + règles d'abort de WS2.
Données : LS-960 + 200–500 h long-form/spontané pseudo-labelisé (teacher offline Qwen3-ASR-1.7B,
tooling existant). Barre de promotion : gate long-form SANS reset < 0.187 human-ref, gates
courts à ±0.02 de WS2.
Pourquoi le décodeur-gelé-CE ne refait pas l'échec D2a : D2a entraînait le DÉCODEUR sur 15 h
(overfit) ; ici le décodeur est gelé, l'encodeur voit ~1.5k h, et le signal vit là où le drift vit.

### Phase 2 — co-adaptation jointe échelle moyenne, 200–600 $, conditionnelle
Seulement si Phase 1 retire le reset mais plafonne > ~0.13. = les « 95 % » de Voxtral en
miniature : encodeur + décodeur-LoRA joints, CE sommée, 1–3k h pseudo-labelisées diverses
(YODAS/GigaSpeech/earnings), régime state-exposure conservé, garde z-loss sur l'équilibre des
normes audio/texte. Pilote 25 $ (50 h, 5k steps) avant d'engager. ~40 % d'atteindre ≤0.12 ;
~20–25 % d'atteindre le point fenêtré 0.084.

### Option écartée — pretrain Voxtral complet : 30k–300k$+, hors budget de 2-3 ordres de grandeur.

### Option parallèle — les poids Voxtral eux-mêmes
Voxtral-Mini-4B-Realtime est DÉJÀ un backend WLK. ~85 % de proba de battre 0.187 sur MCIF,
~50 % d'atteindre ≤0.084, pour ~10 $ d'éval. Coût : 4.4B params au serving (~9 GB bf16 + 2 KV
caches) vs notre 0.6B. C'est l'option « perfs Voxtral garanties » ; le Qwen 0.6B causal reste
l'option légère.

## Arbre de décision
- Sinks récupèrent le no-reset (~15 % proba) → ship sinks, WS4 direct.
- Courbe CE plate → WS3 falsifié → reset + backend Voxtral, stop training.
- Sinon → Phase 1 ; si retire le reset mais plafonne → pilote Phase 2.
- Dépense pire-cas jusqu'à Phase 2 : ~800 $ ; dépense attendue jusqu'à une décision : 100–200 $.
