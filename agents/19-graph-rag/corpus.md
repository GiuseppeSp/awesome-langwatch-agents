# Corpus / knowledge graph — a fictional academic registry

Each bullet is one edge of a knowledge graph (subject → relation → object).
Flat RAG treats these as 24 independent passages and keyword-retrieves the top-k.
Graph RAG parses them into a graph and traverses it: gather the connected
subgraph around the query's entities. The difference only matters when the answer
depends on *structure* — chains, or global counts across many edges that no
top-k of passages can hold at once.

Same registry as #18, expanded so academies have different scholar counts
(Vellum 4, Sere 3, Tarn 2) — which is what makes the global/aggregate questions
real.

- Aldric belongs to the Vellum Academy.
- Doran belongs to the Vellum Academy.
- Faye belongs to the Vellum Academy.
- Iven belongs to the Vellum Academy.
- Beya belongs to the Sere Conservatory.
- Esha belongs to the Sere Conservatory.
- Gale belongs to the Sere Conservatory.
- Corwin belongs to the Tarn Institute.
- Hana belongs to the Tarn Institute.
- The Vellum Academy is located in Threnn.
- The Sere Conservatory is located in Oloss.
- The Tarn Institute is located in Bred.
- Threnn is governed by the Pell Concord.
- Oloss is governed by the Maelor League.
- Bred is governed by the Pell Concord.
- Aldric specializes in astronomy.
- Doran specializes in metallurgy.
- Faye specializes in botany.
- Iven specializes in linguistics.
- Beya specializes in cartography.
- Esha specializes in navigation.
- Gale specializes in rhetoric.
- Corwin specializes in herbalism.
- Hana specializes in geology.
