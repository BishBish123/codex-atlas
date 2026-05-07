# Codex-Atlas eval report

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| Route correctness | 93.3% |
| Citation recall (mean) | 0.13 |
| Citation precision (mean) | 0.06 |
| Faithfulness (mean) | 0.38 |
| Faithfulness (RAGAS) | 0.38 |
| Answer relevancy (RAGAS) | 0.14 |
| Context precision (RAGAS) | 0.35 |
| Context recall (RAGAS) | 0.35 |
| p50 latency (ms) | 2.6 |
| p95 latency (ms) | 4.1 |
| p99 latency (ms) | 5.5 |
| Tool calls / q (mean) | 1.20 |
| Cost estimate (USD, total) | 0.7350 |

## By category

| Category | n | Route acc. | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 2 | 100% | 0.00 | 0.00 |
| import-chain | 3 | 100% | 0.00 | 0.00 |
| lookup | 6 | 100% | 0.00 | 0.00 |
| multi-hop | 4 | 50% | 0.00 | 0.00 |
| neighborhood | 4 | 100% | 0.25 | 0.06 |
| out-of-scope | 2 | 100% | 0.00 | 0.00 |
| structural | 5 | 100% | 0.60 | 0.29 |
| summarization | 4 | 100% | 0.00 | 0.00 |

## Failure taxonomy

| Bucket | Count |
| --- | ---: |
| none | 1 |
| missing_node | 0 |
| wrong_route | 2 |
| hallucinated | 3 |
| partial | 0 |
| ungrounded | 3 |
| off_topic | 21 |

## Per question

| qid | route ok | recall | prec | faith | ms | tools | bucket | preview |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| lookup-1 | yes | 0.00 | 0.00 | 0.00 | 3.2 | 1 | off_topic | # what does Encoder do  ## `codex_atlas.cli.search` (src/codex_atlas/cli.py:428- |
| lookup-2 | yes | 0.00 | 0.00 | 0.00 | 2.8 | 1 | off_topic | # explain how chunk_id is constructed  ## `codex_atlas.agent_langgraph._require_ |
| lookup-3 | yes | 0.00 | 0.00 | 0.00 | 2.9 | 1 | off_topic | # how does the heuristic grader score retrieval  ## `codex_atlas.indexer.ast_par |
| structural-1 | yes | 0.00 | 0.00 | 0.00 | 3.1 | 1 | off_topic | # who calls find_callers  ## `codex_atlas.eval.harness.aggregate` (src/codex_atl |
| structural-2 | yes | 1.00 | 0.40 | 1.00 | 0.8 | 1 | hallucinated | # who calls _extract_qualified_name  ## `codex_atlas.indexer.graph.CallGraph.has |
| structural-3 | yes | 0.00 | 0.00 | 0.00 | 3.1 | 1 | off_topic | # callers of register_async  ## `codex_atlas.judge.calibration._write_calibratio |
| hybrid-1 | no | 0.00 | 0.00 | 0.33 | 2.9 | 1 | wrong_route | # show me all retriever-related code  ## `codex_atlas.retriever.Retriever._neigh |
| hybrid-2 | yes | 0.00 | 0.00 | 0.00 | 6.0 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.indexer.neo4j_graph.Neo4jCallGra |
| summary-1 | yes | 0.00 | 0.00 | 0.00 | 3.8 | 1 | off_topic | # walk me through the agent loop  ## `codex_atlas.eval.ragas.heuristic.ContextRe |
| summary-2 | yes | 0.00 | 0.00 | 0.00 | 4.0 | 1 | off_topic | # overview of the indexer module  ## `codex_atlas.mcp_server._agent` (src/codex_ |
| neighborhood-1 | yes | 1.00 | 0.25 | 1.00 | 0.6 | 1 | hallucinated | # neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers  ## `codex_at |
| neighborhood-2 | yes | 0.00 | 0.00 | 1.00 | 0.7 | 1 | off_topic | # everything around codex_atlas.retriever.Retriever.retrieve  ## `codex_atlas.ag |
| import-chain-1 | yes | 0.00 | 0.00 | 0.00 | 0.8 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| import-chain-2 | yes | 0.00 | 0.00 | 0.00 | 0.7 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-1 | yes | 0.00 | 0.00 | 1.00 | 2.5 | 1 | off_topic | # what does frobnicate_widget do  ## `codex_atlas.store._core.ChunkStoreProtocol |
| oos-1 | yes | 0.00 | 0.00 | 1.00 | 2.7 | 1 | off_topic | # how do I deploy this to AWS Lambda  ## `codex_atlas.mcp_server.search_code` (s |
| lookup-4 | yes | 0.00 | 0.00 | 0.00 | 2.5 | 1 | off_topic | # what does FakeEncoder do  ## `codex_atlas.embed.load_sentence_transformer_enco |
| lookup-5 | yes | 0.00 | 0.00 | 0.00 | 2.4 | 1 | off_topic | # how does InMemoryChunkStore store chunks  ## `codex_atlas.agent_langgraph._sta |
| lookup-6 | yes | 0.00 | 0.00 | 0.00 | 2.5 | 1 | off_topic | # what does the StitchSynthesizer produce  ## `codex_atlas.cli._root` (src/codex |
| structural-4 | yes | 1.00 | 0.67 | 1.00 | 0.6 | 1 | none | # who calls hybrid_score  ## `codex_atlas.retriever.Retriever._hybrid` (src/code |
| structural-5 | yes | 1.00 | 0.40 | 1.00 | 0.7 | 1 | hallucinated | # callees of parse_corpus  ## `codex_atlas.cli._require_corpus_dir_and_parse` (s |
| hybrid-3 | no | 0.00 | 0.00 | 0.00 | 2.5 | 1 | wrong_route | # show me all embed-related code  ## `codex_atlas.store._core.ChunkStoreProtocol |
| hybrid-4 | yes | 0.00 | 0.00 | 0.50 | 4.1 | 1 | off_topic | # find all code that does cache-related work  ## `codex_atlas.store._core.ChunkS |
| summary-3 | yes | 0.00 | 0.00 | 0.00 | 4.0 | 1 | off_topic | # walk me through how the retriever classifies a query  ## `codex_atlas.mcp_serv |
| summary-4 | yes | 0.00 | 0.00 | 0.00 | 3.7 | 1 | off_topic | # overview of the store module  ## `codex_atlas.embed.FakeEncoder.__post_init__` |
| neighborhood-3 | yes | 0.00 | 0.00 | 0.50 | 0.6 | 1 | off_topic | # neighborhood of codex_atlas.indexer.graph.CallGraph.ingest  ## `codex_atlas.ag |
| neighborhood-4 | yes | 0.00 | 0.00 | 1.00 | 1.0 | 1 | off_topic | # everything around codex_atlas.agent.Agent.run  ## `codex_atlas.agent.Agent._an |
| import-chain-3 | yes | 0.00 | 0.00 | 0.00 | 0.8 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-2 | yes | 0.00 | 0.00 | 1.00 | 2.7 | 1 | off_topic | # what does authenticate_user do  ## `codex_atlas.store._core._validate_schema_v |
| oos-2 | yes | 0.00 | 0.00 | 1.00 | 2.6 | 1 | off_topic | # what is the weather in San Francisco today  ## `codex_atlas.judge.calibration. |
