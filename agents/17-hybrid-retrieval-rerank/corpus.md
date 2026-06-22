# Corpus — a fictional herbalist's compendium

Each bullet is one retrievable passage. The entities are invented, so the model
must rely on retrieval. The passages use *formal* names and terms ("Lucentia
herba", "joint inflammation"); the `poor` queries in the dataset use vague,
layperson phrasing ("that glowing moss for achy joints").

Same corpus and dataset as #15 (query-rewriting) and #16 (HyDE), on purpose:
those two transformed the *query* to make lexical retrieval work; this agent
changes the *retriever* (adds semantic search + reranking) instead. Reusing the
identical data lets all three be compared head-to-head.

- Sunmoss, known formally as Lucentia herba, is harvested in the highlands of Veth and is used to treat joint inflammation.
- Gravewort grows in shaded marshes, and its root is brewed into a tonic that reduces fever.
- The Azure Fern releases its spores only at night; alchemists collect them to brew a sleeping draught.
- Emberleaf is a red-veined shrub from the Ashlands; chewing its leaves dulls toothache.
- Tidecap mushrooms grow on coastal rocks and are dried into a remedy for seasickness.
- Whispervine is a climbing plant whose sap, once distilled, sharpens memory and focus.
- The Frostbloom flower blooms in deep winter, and its petals are steeped to soothe a sore throat.
- Goldcrest berries from the Veth highlands are pressed into an oil that heals burns.
