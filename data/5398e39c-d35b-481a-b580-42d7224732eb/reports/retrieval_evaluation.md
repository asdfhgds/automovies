# Retrieval Evaluation

Method: TF-IDF semantic + transcript + dialogue overlap.

Assessment scale: **GOOD** / **PARTIAL** / **WRONG**.

Milestone verdict (real run, project `5398e39c-d35b-481a-b580-42d7224732eb`, window 0-120s:
opening credits into the opening monologue — **Portuguese-dubbed clip**; English is the
normal case): **1 GOOD / 2 PARTIAL / 5 WRONG**. TF-IDF
word-overlap retrieval works when the query's vocabulary is literally present in the
vision fields (Q4, emphasized object), but every narrative/thematic query fails or is
noisy -- there is no semantic reasoning, only token overlap.

| # | Query | Top scenes (timestamps) | Scores | Human assessment |
|---|-------|-------------------------|--------|------------------|
| 1 | Find a scene where one character appears to have a choice but another character  | scene-6 (21.521522-23.690357s), scene-18 (48.25659-51.384718s), scene-27 (71.321321-80.33033s), scene-25 (63.938939-65.7 | 0.1245, 0.057, 0.0558, 0.0536, 0.0524 | WRONG |
| 2 | Find the strongest moment of tension between two characters. | scene-4 (18.309977-19.436103s), scene-29 (86.211211-95.637304s), scene-14 (40.165165-41.791792s), scene-30 (95.637304-10 | 0.0696, 0.0446, 0.0421, 0.0338, 0.0312 | PARTIAL |
| 3 | Find a scene where the meaning comes mostly from what characters do rather than  | scene-10 (30.780781-33.700367s), scene-8 (25.275275-26.860194s), scene-19 (51.384718-54.095762s), scene-17 (46.129463-48 | 0.0699, 0.0422, 0.035, 0.0284, 0.0235 | WRONG |
| 4 | Find a moment where an important object is visually emphasized. | scene-25 (63.938939-65.732399s), scene-15 (41.791792-45.378712s), scene-5 (19.436103-21.521522s), scene-4 (18.309977-19. | 0.1555, 0.0722, 0.0614, 0.0332, 0.027 | GOOD |
| 5 | Find a scene that demonstrates the protagonist's relationship with fate. | scene-8 (25.275275-26.860194s), scene-19 (51.384718-54.095762s), scene-17 (46.129463-48.25659s), scene-25 (63.938939-65. | 0.0811, 0.0673, 0.0546, 0.0451, 0.0445 | WRONG |
| 6 | Find a scene where a character's behavior contradicts what they say. | scene-10 (30.780781-33.700367s), scene-8 (25.275275-26.860194s), scene-31 (101.810143-104.020687s), scene-19 (51.384718- | 0.0531, 0.0321, 0.0313, 0.0266, 0.0216 | WRONG |
| 7 | Who is present in this scene and what are they doing? | scene-8 (25.275275-26.860194s), scene-19 (51.384718-54.095762s), scene-17 (46.129463-48.25659s), scene-25 (63.938939-65. | 0.0811, 0.0673, 0.0546, 0.0451, 0.0445 | PARTIAL |
| 8 | When does the most important visual event occur? | scene-15 (41.791792-45.378712s), scene-5 (19.436103-21.521522s), scene-3 (16.975309-18.309977s) | 0.0779, 0.0663, 0.0484 | WRONG |

## Human notes

1. **WRONG** — Top hit scene-6 (21.52-23.69s) and the rest of the list are
   revolver/barman close-ups in the opening credits into the narrator's opening
   monologue. No scene in this window shows one character making a choice while
   another controls the situation. TF-IDF matched generic words (choice / control /
   character) in the vision fields without any narrative reasoning; all scores sit
   near the floor (0.05-0.12).
2. **PARTIAL** — scene-4 (18.31s revolver close-up) and scene-29 (86.21-95.64s bar
   monologue) rank high, and scene-30 (95.64-101.81s, the narrator with the gun at
   his mouth -- arguably the tensest beat in the window) appears 4th at 0.0338.
   Genuinely tense scenes do surface, but the strongest is not ranked first and the
   score gap is thin (0.0696 -> 0.0312).
3. **WRONG** — Top hits scene-10 (30.78s) and scene-8 (25.28s) are
   dialogue/monologue face-and-revolver close-ups where the meaning arrives from
   speech, not action. The window (title sequence + opening monologue) has no strong
   action-over-dialogue beat, so the query with no shared vocabulary scores at the
   floor.
4. **GOOD** — scene-25 (63.94-65.73s, a glowing blue object in extreme close-up,
   0.1555 -- the highest single score in the whole eval), scene-15 (41.79s) and
   scene-4 (18.31s revolver close-up) all center on the revolver/object the
   cinematography actually emphasizes. This is the one query where object emphasis
   genuinely matches on-screen framing.
5. **WRONG** — Top hits scene-8 (25.28s), scene-19 (51.38s), scene-17 (46.13s) are
   monologue/revolver fragments; nothing in this window dramatizes a relationship
   with fate. Thematic query with no shared vocabulary, so TF-IDF returns
   floor-scoring word-overlap noise.
6. **WRONG** — scene-10 (30.78s), scene-8 (25.28s), scene-31 (101.81s) are
   speaking/face shots whose transcript affirms what the character does (e.g. "can't
   talk with a gun in your mouth"); there is no contradiction beat in the window.
   Word-level overlap produces this ranking, not understanding of intent.
7. **PARTIAL** — The richest who-is-present-and-doing-what content (scene-30 man
   holding a gun to his mouth, scene-31 blood visible on the face) exists in the
   index, but the rankings (scene-8/19/17/25) reward token overlap with the credits
   close-ups. The information is present in the vision fields; the ranking is
   keyword-driven and misses it.
8. **WRONG** — The most important visual event in the window is the gun-to-mouth
   reveal (~95.6-104s, scenes 30-31); retrieval returned scene-15 (41.79s), scene-5
   (19.44s), scene-3 (16.98s logo). Also underlines the temporal weakness: a "when"
   query gets no time-anchored answer with lexicon-based retrieval.