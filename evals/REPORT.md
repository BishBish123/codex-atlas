# Codex-Atlas eval report

## Headline

| Metric | Value |
| --- | ---: |
| Questions | 16 |
| Route correctness | 93.8% |
| Citation recall (mean) | 0.22 |
| Citation precision (mean) | 0.06 |
| p50 latency (ms) | 7.3 |
| p95 latency (ms) | 15.4 |
| p99 latency (ms) | 15.7 |
| Tool calls / q (mean) | 2.00 |
| Cost estimate (USD, total) | 0.3144 |

## By category

| Category | n | Route acc. | Recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| failure-likely | 1 | 100% | 0.00 | 0.00 |
| import-chain | 2 | 100% | 0.00 | 0.00 |
| lookup | 3 | 100% | 0.00 | 0.00 |
| multi-hop | 2 | 50% | 0.00 | 0.00 |
| neighborhood | 2 | 100% | 0.50 | 0.12 |
| out-of-scope | 1 | 100% | 0.00 | 0.00 |
| structural | 3 | 100% | 0.67 | 0.24 |
| summarization | 2 | 100% | 0.25 | 0.02 |

## Failure taxonomy

| Bucket | Count |
| --- | ---: |
| none | 0 |
| missing_node | 0 |
| wrong_route | 1 |
| hallucinated | 3 |
| partial | 1 |
| ungrounded | 2 |
| off_topic | 9 |

## Per question

| qid | route ok | recall | prec | ms | tools | bucket | preview |
| --- | :---: | ---: | ---: | ---: | ---: | --- | --- |
| lookup-1 | yes | 0.00 | 0.00 | 12.0 | 3 | off_topic | # what does Encoder do (in this codebase, with code-level detail)  ## `codex_atl |
| lookup-2 | yes | 0.00 | 0.00 | 10.2 | 3 | off_topic | # explain how chunk_id is constructed (in this codebase, with code-level detail) |
| lookup-3 | yes | 0.00 | 0.00 | 10.7 | 3 | off_topic | # how does the heuristic grader score retrieval (in this codebase, with code-lev |
| structural-1 | yes | 1.00 | 0.33 | 2.0 | 1 | hallucinated | # who calls find_callers  ## `codex_atlas.indexer.graph.CallGraph._traverse` (sr |
| structural-2 | yes | 1.00 | 0.40 | 1.5 | 1 | hallucinated | # who calls _extract_qualified_name  ## `codex_atlas.indexer.graph.CallGraph.has |
| structural-3 | yes | 0.00 | 0.00 | 6.2 | 1 | off_topic | # callers of register_async  ## `codex_atlas.indexer.ast_parser._Collector._hand |
| hybrid-1 | no | 0.00 | 0.00 | 15.1 | 3 | wrong_route | # show me all retriever-related code (in this codebase, with code-level detail)  |
| hybrid-2 | yes | 0.00 | 0.00 | 15.2 | 1 | off_topic | # end-to-end indexing pipeline  ## `codex_atlas.cli.ask._run` (src/codex_atlas/c |
| summary-1 | yes | 0.50 | 0.04 | 6.2 | 1 | partial | # walk me through the agent loop  ## `codex_atlas.store.InMemoryChunkStore.fetch |
| summary-2 | yes | 0.00 | 0.00 | 8.3 | 1 | off_topic | # overview of the indexer module  ## `codex_atlas.mcp_server._graph` (src/codex_ |
| neighborhood-1 | yes | 1.00 | 0.25 | 3.3 | 1 | hallucinated | # neighborhood of codex_atlas.indexer.graph.CallGraph.find_callers  ## `codex_at |
| neighborhood-2 | yes | 0.00 | 0.00 | 2.7 | 1 | off_topic | # everything around codex_atlas.retriever.Retriever.retrieve  ## `codex_atlas.ag |
| import-chain-1 | yes | 0.00 | 0.00 | 0.7 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| import-chain-2 | yes | 0.00 | 0.00 | 0.5 | 3 | ungrounded | I could not find any relevant code chunks for that query in the indexed corpus.  |
| refusal-1 | yes | 0.00 | 0.00 | 15.8 | 3 | off_topic | # what does frobnicate_widget do (in this codebase, with code-level detail)  ##  |
| oos-1 | yes | 0.00 | 0.00 | 15.3 | 3 | off_topic | # how do I deploy this to AWS Lambda (in this codebase, with code-level detail)  |
