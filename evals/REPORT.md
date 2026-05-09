# Codex-Atlas eval report

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| Route correctness | 93.3% |
| Citation recall (mean) | 0.13 |
| Citation precision (mean) | 0.06 |
| Faithfulness (mean) | 0.38 |
| Faithfulness (RAGAS) | 0.36 |
| Answer relevancy (RAGAS) | 0.19 |
| Context precision (RAGAS) | 0.35 |
| Context recall (RAGAS) | 0.35 |
| p50 latency (ms) | 230.1 |
| p95 latency (ms) | 955.6 |
| p99 latency (ms) | 1186.2 |
| Tool calls / q (mean) | 1.20 |
| Cost estimate (USD, total) | 0.6379 |

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
| lookup-1 | yes | 0.00 | 0.00 | 0.00 | 1272.1 | 1 | off_topic | The Encoder is used at index time and is resolved using the `_resolve_encoder` f |
| lookup-2 | yes | 0.00 | 0.00 | 0.00 | 895.8 | 1 | off_topic | The code does not explicitly show how `chunk_id` is constructed. However, it is  |
| lookup-3 | yes | 0.00 | 0.00 | 0.00 | 975.8 | 1 | off_topic | The heuristic grader scores retrieval based on a fallback mechanism when the LLM |
| structural-1 | yes | 0.00 | 0.00 | 0.00 | 594.7 | 1 | off_topic | The function `find_callers` is not present in the provided code chunks. Therefor |
| structural-2 | yes | 1.00 | 0.40 | 1.00 | 653.2 | 1 | hallucinated |  `_extract_qualified_name` is called by `codex_atlas.retriever.Retriever._import |
| structural-3 | yes | 0.00 | 0.00 | 0.00 | 416.4 | 1 | off_topic | # callers of register_async  ## `codex_atlas.judge.calibration._write_calibratio |
| hybrid-1 | no | 0.00 | 0.00 | 0.33 | 231.7 | 1 | wrong_route | # show me all retriever-related code  ## `codex_atlas.retriever.Retriever._neigh |
| hybrid-2 | yes | 0.00 | 0.00 | 0.00 | 228.4 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.indexer.neo4j_graph.Neo4jCallGra |
| summary-1 | yes | 0.00 | 0.00 | 0.00 | 232.8 | 1 | off_topic | # walk me through the agent loop  ## `codex_atlas.eval.ragas.heuristic.ContextRe |
| summary-2 | yes | 0.00 | 0.00 | 0.00 | 233.5 | 1 | off_topic | # overview of the indexer module  ## `codex_atlas.mcp_server._agent` (src/codex_ |
| neighborhood-1 | yes | 1.00 | 0.25 | 1.00 | 183.4 | 1 | hallucinated | # neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers  ## `codex_at |
| neighborhood-2 | yes | 0.00 | 0.00 | 1.00 | 195.7 | 1 | off_topic | # everything around codex_atlas.retriever.Retriever.retrieve  ## `codex_atlas.ag |
| import-chain-1 | yes | 0.00 | 0.00 | 0.00 | 339.0 | 3 | ungrounded | There is no code provided to analyze. Please provide the code chunks to determin |
| import-chain-2 | yes | 0.00 | 0.00 | 0.00 | 930.9 | 3 | ungrounded | To determine the import chain for [ungrounded: codex_atlas.indexer.graph], we ne |
| refusal-1 | yes | 0.00 | 0.00 | 1.00 | 163.4 | 1 | off_topic | # what does frobnicate_widget do  ## `codex_atlas.store._core.ChunkStoreProtocol |
| oos-1 | yes | 0.00 | 0.00 | 1.00 | 174.9 | 1 | off_topic | # how do I deploy this to AWS Lambda  ## `codex_atlas.mcp_server.search_code` (s |
| lookup-4 | yes | 0.00 | 0.00 | 0.00 | 146.8 | 1 | off_topic | # what does FakeEncoder do  ## `codex_atlas.embed.load_sentence_transformer_enco |
| lookup-5 | yes | 0.00 | 0.00 | 0.00 | 168.6 | 1 | off_topic | # how does InMemoryChunkStore store chunks  ## `codex_atlas.agent_langgraph._sta |
| lookup-6 | yes | 0.00 | 0.00 | 0.00 | 191.7 | 1 | off_topic | # what does the StitchSynthesizer produce  ## `codex_atlas.cli._root` (src/codex |
| structural-4 | yes | 1.00 | 0.67 | 1.00 | 134.1 | 1 | none | # who calls hybrid_score  ## `codex_atlas.retriever.Retriever._hybrid` (src/code |
| structural-5 | yes | 1.00 | 0.40 | 1.00 | 141.9 | 1 | hallucinated | # callees of parse_corpus  ## `codex_atlas.cli._require_corpus_dir_and_parse` (s |
| hybrid-3 | no | 0.00 | 0.00 | 0.00 | 173.3 | 1 | wrong_route | # show me all embed-related code  ## `codex_atlas.store._core.ChunkStoreProtocol |
| hybrid-4 | yes | 0.00 | 0.00 | 0.50 | 210.0 | 1 | off_topic | # find all code that does cache-related work  ## `codex_atlas.store._core.ChunkS |
| summary-3 | yes | 0.00 | 0.00 | 0.00 | 211.3 | 1 | off_topic | # walk me through how the retriever classifies a query  ## `codex_atlas.mcp_serv |
| summary-4 | yes | 0.00 | 0.00 | 0.00 | 238.1 | 1 | off_topic | # overview of the store module  ## `codex_atlas.embed.FakeEncoder.__post_init__` |
| neighborhood-3 | yes | 0.00 | 0.00 | 0.50 | 270.4 | 1 | off_topic | # neighborhood of codex_atlas.indexer.graph.CallGraph.ingest  ## `codex_atlas.ag |
| neighborhood-4 | yes | 0.00 | 0.00 | 1.00 | 173.7 | 1 | off_topic | # everything around codex_atlas.agent.Agent.run  ## `codex_atlas.agent.Agent._an |
| import-chain-3 | yes | 0.00 | 0.00 | 0.00 | 450.0 | 3 | ungrounded | There are no code chunks provided to analyze which modules import [ungrounded: c |
| refusal-2 | yes | 0.00 | 0.00 | 1.00 | 247.5 | 1 | off_topic | # what does authenticate_user do  ## `codex_atlas.store._core._validate_schema_v |
| oos-2 | yes | 0.00 | 0.00 | 1.00 | 163.2 | 1 | off_topic | # what is the weather in San Francisco today  ## `codex_atlas.judge.calibration. |
