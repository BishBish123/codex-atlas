# Codex-Atlas eval report

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 30 |
| Route correctness | 93.3% |
| Citation recall (mean) | 0.28 |
| Citation precision (mean) | 0.08 |
| Faithfulness (mean) | 0.64 |
| Faithfulness (RAGAS) | n/a |
| Answer relevancy (RAGAS) | n/a |
| Context precision (RAGAS) | n/a |
| Context recall (RAGAS) | n/a |
| p50 latency (ms) | 184.9 |
| p95 latency (ms) | 563.1 |
| p99 latency (ms) | 798.7 |
| Tool calls / q (mean) | 1.20 |
| Cost estimate (USD, total) | 0.8185 |

## By category

| Category | n | Route acc. | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 2 | 100% | 0.00 | 0.00 |
| import-chain | 3 | 100% | 0.00 | 0.00 |
| lookup | 6 | 100% | 0.33 | 0.04 |
| multi-hop | 4 | 50% | 0.00 | 0.00 |
| neighborhood | 4 | 100% | 0.25 | 0.06 |
| out-of-scope | 2 | 100% | 0.00 | 0.00 |
| structural | 5 | 100% | 0.80 | 0.34 |
| summarization | 4 | 100% | 0.38 | 0.04 |

## Failure taxonomy

| Bucket | Count |
| --- | ---: |
| none | 1 |
| missing_node | 0 |
| wrong_route | 2 |
| hallucinated | 7 |
| partial | 1 |
| ungrounded | 3 |
| off_topic | 16 |

## Per question

| qid | route ok | recall | prec | faith | ms | tools | bucket | preview |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| lookup-1 | yes | 0.00 | 0.00 | 1.00 | 260.7 | 1 | off_topic | # what does Encoder do  ## `codex_atlas.retriever.Retriever.__init__` (src/codex |
| lookup-2 | yes | 1.00 | 0.12 | 1.00 | 180.0 | 1 | hallucinated | # explain how chunk_id is constructed  ## `codex_atlas.indexer.ast_parser.Chunk. |
| lookup-3 | yes | 1.00 | 0.12 | 1.00 | 181.1 | 1 | hallucinated | # how does the heuristic grader score retrieval  ## `codex_atlas.agent.Agent._gr |
| structural-1 | yes | 1.00 | 0.25 | 1.00 | 249.4 | 1 | hallucinated | # who calls find_callers  ## `codex_atlas.mcp_server.find_callers` (src/codex_at |
| structural-2 | yes | 1.00 | 0.40 | 1.00 | 30.7 | 1 | hallucinated | # who calls _extract_qualified_name  ## `codex_atlas.indexer.graph.CallGraph.has |
| structural-3 | yes | 0.00 | 0.00 | 0.00 | 851.5 | 1 | off_topic | # callers of register_async  ## `codex_atlas.store._core.ChunkStore._connect` (s |
| hybrid-1 | no | 0.00 | 0.00 | 0.33 | 669.4 | 1 | wrong_route | # show me all retriever-related code  ## `codex_atlas.agent.Agent.__init__` (src |
| hybrid-2 | yes | 0.00 | 0.00 | 0.00 | 263.3 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.cli.index` (src/codex_atlas/cli. |
| summary-1 | yes | 0.50 | 0.05 | 0.50 | 229.0 | 1 | partial | # walk me through the agent loop  ## `codex_atlas.mcp_server._agent_config` (src |
| summary-2 | yes | 0.00 | 0.00 | 0.67 | 195.9 | 1 | off_topic | # overview of the indexer module  ## `codex_atlas.cli.index` (src/codex_atlas/cl |
| neighborhood-1 | yes | 1.00 | 0.25 | 1.00 | 3.3 | 1 | hallucinated | # neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers  ## `codex_at |
| neighborhood-2 | yes | 0.00 | 0.00 | 1.00 | 2.3 | 1 | off_topic | # everything around codex_atlas.retriever.Retriever.retrieve  ## `codex_atlas.ag |
| import-chain-1 | yes | 0.00 | 0.00 | 0.00 | 7.7 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| import-chain-2 | yes | 0.00 | 0.00 | 0.00 | 5.0 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-1 | yes | 0.00 | 0.00 | 1.00 | 433.1 | 1 | off_topic | # what does frobnicate_widget do  ## `codex_atlas.indexer.ast_parser._Collector. |
| oos-1 | yes | 0.00 | 0.00 | 1.00 | 183.8 | 1 | off_topic | # how do I deploy this to AWS Lambda  ## `codex_atlas.synthesis.groq.GroqSynthes |
| lookup-4 | yes | 0.00 | 0.00 | 1.00 | 186.2 | 1 | off_topic | # what does FakeEncoder do  ## `codex_atlas.cli._resolve_encoder` (src/codex_atl |
| lookup-5 | yes | 0.00 | 0.00 | 1.00 | 221.6 | 1 | off_topic | # how does InMemoryChunkStore store chunks  ## `codex_atlas.store._core._load_ch |
| lookup-6 | yes | 0.00 | 0.00 | 0.00 | 193.1 | 1 | off_topic | # what does the StitchSynthesizer produce  ## `codex_atlas.synthesis.groq.GroqSy |
| structural-4 | yes | 1.00 | 0.67 | 1.00 | 1.3 | 1 | none | # who calls hybrid_score  ## `codex_atlas.retriever.Retriever._hybrid` (src/code |
| structural-5 | yes | 1.00 | 0.40 | 1.00 | 4.6 | 1 | hallucinated | # callees of parse_corpus  ## `codex_atlas.cli._require_corpus_dir_and_parse` (s |
| hybrid-3 | no | 0.00 | 0.00 | 0.00 | 207.7 | 1 | wrong_route | # show me all embed-related code  ## `codex_atlas.store.pgvector.PgVectorChunkSt |
| hybrid-4 | yes | 0.00 | 0.00 | 0.00 | 199.6 | 1 | off_topic | # find all code that does cache-related work  ## `codex_atlas.mcp_server._graph` |
| summary-3 | yes | 1.00 | 0.10 | 1.00 | 186.0 | 1 | hallucinated | # walk me through how the retriever classifies a query  ## `codex_atlas.agent_la |
| summary-4 | yes | 0.00 | 0.00 | 0.33 | 168.8 | 1 | off_topic | # overview of the store module  ## `codex_atlas.mcp_server._store_backend` (src/ |
| neighborhood-3 | yes | 0.00 | 0.00 | 0.50 | 0.8 | 1 | off_topic | # neighborhood of codex_atlas.indexer.graph.CallGraph.ingest  ## `codex_atlas.ag |
| neighborhood-4 | yes | 0.00 | 0.00 | 1.00 | 1.4 | 1 | off_topic | # everything around codex_atlas.agent.Agent.run  ## `codex_atlas.agent.Agent._an |
| import-chain-3 | yes | 0.00 | 0.00 | 0.00 | 4.0 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-2 | yes | 0.00 | 0.00 | 1.00 | 166.4 | 1 | off_topic | # what does authenticate_user do  ## `codex_atlas.synthesis.groq._build_user_mes |
| oos-2 | yes | 0.00 | 0.00 | 1.00 | 187.8 | 1 | off_topic | # what is the weather in San Francisco today  ## `codex_atlas.judge.llm.LLMJudge |
