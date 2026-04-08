---
title: "RAGの検索をAIに任せたら精度が79%上がった"
emoji: "🔍"
type: "tech"
topics: ["RAG", "LLM", "Python", "AI", "機械学習"]
published: true
---

# RAGの検索をAIに任せたら精度が79%上がった

RAG（Retrieval-Augmented Generation）の検索パイプラインは、ほとんどの場合こう組まれている:

```
クエリ → ベクトル検索 → Top-K取得 → LLMに全部渡す
```

この固定パイプラインこそが、RAGの精度を制限している元凶だった。

2026年2月のArXiv論文（arXiv:2602.03442）がA-RAG（Agentic RAG）を提案し、検索パイプラインをAIエージェントに置き換えた。結果: **マルチホップ質問応答の精度が79%向上**（50.2% → 89.7%）。しかも検索トークン数は半分以下に減った。

精度が上がって検索量が減る。直感に反するこの結果の仕組みを分解する。

---

## 固定パイプラインRAGの3つの限界

従来のRAGが苦手なタスクは明確だ。

### 限界1: マルチホップ質問に弱い

```
質問: 「Xを発明した人物の出身大学はどこか？」

必要な検索:
  1回目: 「Xを発明した人物」→ 人物名を特定
  2回目: 「その人物の出身大学」→ 大学名を取得

固定パイプライン:
  1回のベクトル検索で「Xの発明者 + 出身大学」を取ろうとする
  → 直接答えを含むチャンクがない
  → 関連性の低いチャンクを大量に取得
  → LLMが推測で回答 → 不正確
```

マルチホップ質問は実用上かなりの割合を占める。1回の検索で答えが出ない質問に対し、固定パイプラインは構造的に弱い。

### 限界2: 検索粒度が固定

```
Top-K=5の場合:
  質問が単純でも5チャンク取得 → トークンの無駄
  質問が複雑でも5チャンク取得 → 情報不足

必要な粒度は質問ごとに異なる:
  「GPT-4のパラメータ数は？」→ 1チャンクで十分
  「GPT-4と Claude 3.5の長文処理の違いは？」→ 10チャンク必要
```

### 限界3: 検索戦略が固定

```
ベクトル検索のみ:
  意味的に類似したチャンクを取得
  → キーワード完全一致が必要なケース（型番、固有名詞）に弱い

キーワード検索のみ:
  完全一致・部分一致でチャンクを取得
  → 同義語・言い換えに弱い

ハイブリッド検索（固定比率）:
  ベクトル70% + キーワード30% のような固定重み
  → 質問の性質に応じた動的調整ができない
```

---

## A-RAGのアーキテクチャ: 検索をエージェントに任せる

A-RAGの核心は、検索パイプラインを固定フローからエージェントの自律判断に置き換えたこと。

```
従来RAG:
  クエリ → [固定パイプライン] → チャンク → LLM → 回答
  検索方法、粒度、回数がすべて事前に固定

A-RAG:
  クエリ → [エージェント] → 回答
  エージェントが以下を自律的に判断:
    - どの検索ツールを使うか
    - 何回検索するか
    - どの粒度で情報を取得するか
    - いつ検索を止めるか
```

### 3つの検索インターフェース

A-RAGはエージェントに3つのツールを提供する:

```python
# A-RAGの3つの検索ツール
class ARAGTools:
    def keyword_search(self, query: str) -> list[str]:
        """キーワードベースの検索
        用途: 固有名詞、型番、正確な用語の検索"""
        pass

    def semantic_search(self, query: str) -> list[str]:
        """意味ベースのベクトル検索
        用途: 概念的な類似性、言い換え対応"""
        pass

    def chunk_read(self, doc_id: str, chunk_range: str) -> str:
        """特定チャンクの精読
        用途: 検索結果の深掘り、周辺コンテキスト取得"""
        pass
```

エージェントは質問に応じて、これらのツールを自由に組み合わせる。

### マルチホップ質問の処理例

```
質問: 「Transformerを提案した論文の第一著者の現所属は？」

エージェントの行動:
  Step 1: keyword_search("Transformer paper original authors")
    → "Attention Is All You Need", Vaswani et al., 2017
  Step 2: semantic_search("Ashish Vaswani current affiliation 2026")
    → チャンク3件取得
  Step 3: chunk_read(doc_id="result_2", range="full")
    → 詳細情報を精読
  Step 4: 回答生成 → "Essential AI (2023年設立のスタートアップ)"

固定パイプラインなら:
  vector_search("Transformerを提案した論文の第一著者の現所属")
  → 1回の検索で直接答えが出る可能性は低い
  → 「Attention Is All You Need」の内容ばかり取得
  → 2017年時点のGoogle所属を回答するリスク
```

エージェントは2段階の検索を自律的に実行し、各ステップで最適なツールを選択した。固定パイプラインには不可能な動作。

---

## ベンチマーク結果: 数字で見るA-RAGの効果

論文の実験結果（Table 1）から主要データを抜粋。

### GPT-4o-miniバックエンド

| ベンチマーク | Naive RAG | A-RAG | 改善率 |
|-------------|-----------|-------|--------|
| MuSiQue | 38.6% | 46.1% | +19% |
| HotpotQA | 74.5% | 77.1% | +3.5% |
| 2WikiMultiHopQA | 42.6% | 60.2% | +41% |

### GPT-5-miniバックエンド

| ベンチマーク | Naive RAG | A-RAG | 改善率 |
|-------------|-----------|-------|--------|
| MuSiQue | 52.8% | 74.1% | +40% |
| HotpotQA | 81.2% | 94.5% | +16% |
| 2WikiMultiHopQA | 50.2% | 89.7% | **+79%** |

### パターン分析

```python
# 結果から読み取れるパターン
patterns = {
    "multi_hop_improvement": {
        "2Wiki": "+41% (4o-mini) / +79% (5-mini)",
        "MuSiQue": "+19% (4o-mini) / +40% (5-mini)",
        "insight": "マルチホップ質問ほど改善が大きい"
    },
    "model_scaling": {
        "4o_mini_avg": "+21%",
        "5_mini_avg": "+45%",
        "insight": "強いモデルほどA-RAGの恩恵が大きい"
    },
    "single_hop": {
        "HotpotQA": "+3.5% (4o-mini) / +16% (5-mini)",
        "insight": "シングルホップでも改善するが幅は小さい"
    },
    "graphrag_comparison": {
        "HotpotQA_graphrag_4o_mini": "33.2%",
        "HotpotQA_graphrag_5_mini": "82.5%",
        "HotpotQA_naive_rag": "74.5% / 81.2%",
        "insight": "GraphRAGはモデル性能に極度に依存。弱いモデルでは崩壊する"
    }
}
```

3つの重要な知見:
1. **マルチホップ質問で圧倒的改善**: 2WikiMultiHopQAで+79%。固定パイプラインが最も弱い領域でA-RAGが最も強い
2. **モデルが強いほど効果が大きい**: GPT-5-miniの方がGPT-4o-miniより改善率が高い。エージェントの検索判断能力がモデル性能に依存するため
3. **GraphRAGはモデル性能に極度に依存**: GPT-4o-miniではHotpotQA 33.2%（Naive RAGの半分以下）だが、GPT-5-miniでは82.5%でNaive RAGを上回る。弱いモデルでのGraphRAGは危険

### トークン効率

```
HotpotQA (GPT-5-mini):
  Naive RAG: 5,358トークン取得 → 81.2%精度
  A-RAG:     2,737トークン取得 → 94.5%精度

取得トークン数: -49%
精度: +16%
```

少ないトークンで高い精度。エージェントが必要な情報だけを選択的に取得するため、ノイズが減りLLMの回答品質が上がる。これはAPIコストにも直結する。

---

## ローカルLLMでAgentic RAGは動くか

A-RAGの論文はGPT-4o-miniとGPT-5-miniで実験している。ローカルLLMでは?

### 構造的課題

```python
# A-RAGのエージェント動作に必要な能力
agent_requirements = {
    "tool_use": "ツール呼び出し (function calling)",
    "planning": "複数ステップの計画立案",
    "reflection": "検索結果の評価と次の行動決定",
    "context_management": "取得済み情報の保持と統合",
}

# ローカルLLMの対応状況 (RTX 4060 8GB)
local_llm_capability = {
    "Qwen2.5-32B Q4_K_M": {
        "tool_use": "対応 (ChatML format)",
        "planning": "中程度 (単純な2-3ステップ)",
        "reflection": "限定的 (深い評価は困難)",
        "speed": "~10 t/s (ngl=24)",
        "verdict": "単純なAgentic RAGは可能、複雑なマルチホップは困難"
    },
    "Qwen3.5-9B Q4_K_M": {
        "tool_use": "対応",
        "planning": "中程度",
        "reflection": "限定的",
        "speed": "~33 t/s",
        "verdict": "高速だが知識不足。検索判断の質が落ちる可能性"
    }
}
```

### 最小構成の実装

```python
# ローカルLLM + ChromaDB で最小 Agentic RAG
import chromadb
import subprocess
import json

class LocalAgenticRAG:
    def __init__(self, db_path: str, model_path: str):
        self.chroma = chromadb.PersistentClient(
            path=db_path
        )
        self.collection = self.chroma.get_collection("papers")
        self.model = model_path

    def keyword_search(self, query: str, k: int = 5) -> list:
        """ChromaDB の where フィルタでキーワード検索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where_document={"$contains": query.split()[0]}
        )
        return results["documents"][0]

    def semantic_search(self, query: str, k: int = 5) -> list:
        """ChromaDB のベクトル検索"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results["documents"][0]

    def chunk_read(self, doc_id: str) -> str:
        """特定ドキュメントの全文取得"""
        result = self.collection.get(ids=[doc_id])
        return result["documents"][0]

    def agent_query(self, question: str) -> str:
        """エージェントループ: 検索→評価→追加検索→回答"""
        tools_desc = """Available tools:
        - keyword_search(query): keyword-based search
        - semantic_search(query): vector similarity search
        - chunk_read(doc_id): read full document"""

        context = []
        for step in range(3):  # 最大3ステップ
            ctx_str = json.dumps(context, ensure_ascii=False)[:2000]
            prompt = f"""Question: {question}
{tools_desc}
Retrieved so far: {ctx_str}
Decide: call a tool OR answer directly.
Format: TOOL:tool_name(args) or ANSWER:your answer"""

            response = self._llm_call(prompt)

            if response.startswith("ANSWER:"):
                return response[7:]
            elif response.startswith("TOOL:"):
                tool_result = self._execute_tool(response[5:])
                context.append(tool_result)

        return self._llm_call(
            f"Based on: {json.dumps(context, ensure_ascii=False)}\nAnswer: {question}"
        )

    def _execute_tool(self, tool_call: str) -> str:
        """ツール呼び出しをパースして実行"""
        if tool_call.startswith("keyword_search"):
            query = tool_call.split("(", 1)[1].rstrip(")")
            return str(self.keyword_search(query.strip("'\"")))
        elif tool_call.startswith("semantic_search"):
            query = tool_call.split("(", 1)[1].rstrip(")")
            return str(self.semantic_search(query.strip("'\"")))
        elif tool_call.startswith("chunk_read"):
            doc_id = tool_call.split("(", 1)[1].rstrip(")")
            return self.chunk_read(doc_id.strip("'\""))
        return ""

    def _llm_call(self, prompt: str) -> str:
        result = subprocess.run(
            ["llama-cli", "-m", self.model,
             "-p", prompt, "-n", "200", "--temp", "0.1"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
```

この最小構成は動く。ただし論文のGPT-5-miniの+79%改善は期待できない。ローカルLLMのtool_use能力が制限要因になる。

### 現実的な期待値

```python
# ローカルLLM Agentic RAG の期待改善率 (推定)
expected_improvement = {
    "32B_model": {
        "multi_hop": "+15-25% (論文の1/3程度)",
        "single_hop": "+3-5%",
        "reason": "tool_use能力は中程度、planningが弱い"
    },
    "9B_model": {
        "multi_hop": "+5-10%",
        "single_hop": "+1-3%",
        "reason": "検索判断の質が低い。Naive RAGとの差が小さい"
    },
    "recommendation": "32B以上でないとAgentic RAGの恩恵は小さい"
}
```

---

## Agentic RAGを試す前に確認すべきこと

A-RAGは魅力的だが、すべてのRAGユースケースに必要なわけではない。

```
Agentic RAGが有効なケース:
  ✓ マルチホップ質問が多い (調査、リサーチ)
  ✓ 知識ベースが大規模 (1000チャンク以上)
  ✓ 質問の複雑度がバラバラ
  ✓ 精度が最優先 (医療、法律)

Naive RAGで十分なケース:
  ✓ シングルホップ質問が中心 (FAQ、マニュアル検索)
  ✓ 知識ベースが小規模 (100チャンク以下)
  ✓ 質問パターンが均一
  ✓ レイテンシが最優先
```

Agentic RAGのコスト:
- **レイテンシ増加**: 複数回の検索 + LLM呼び出しでNaive RAGの2-5倍
- **トークン消費増加**: エージェントの思考プロセスがトークンを消費（検索トークンは減るがLLM呼び出し回数が増える）
- **実装複雑度**: tool_use対応のプロンプト設計、エラーハンドリング

### コスト構造の比較

```python
# 1質問あたりのコスト比較
cost_comparison = {
    "naive_rag": {
        "llm_calls": 1,
        "search_calls": 1,
        "avg_tokens": 5358,
        "latency": "1-2秒 (API) / 5-10秒 (ローカル32B)",
    },
    "agentic_rag": {
        "llm_calls": "2-4",
        "search_calls": "2-5",
        "avg_tokens": 2737,
        "latency": "3-8秒 (API) / 15-40秒 (ローカル32B)",
    },
}
# 精度+79%の対価: レイテンシ3-4倍
# ROI: マルチホップ質問の頻度が30%以上なら導入検討の価値あり
```

---

## RAGの次の進化はエージェントにある

この記事の要点:

1. **固定パイプラインRAGの限界は検索設計にある**: マルチホップ質問、粒度固定、戦略固定の3つが精度を制限
2. **A-RAGは検索をエージェントに任せる**: 3つのツール（keyword/semantic/chunk_read）をモデルが自律選択
3. **マルチホップ質問で+79%改善**: 固定パイプラインが最も弱い領域で最大効果
4. **少ないトークンで高精度**: 検索トークン-49%、精度+16%
5. **モデルが強いほど効果が大きい**: ローカルLLMでは改善幅が縮む

RAGの進化は、検索設計の人間による最適化から、モデル自身による検索判断へ移行しつつある。固定パイプラインは安定していて予測可能だが、質問の多様性に対応できない。エージェントは不安定だが適応的。

自分のRAGシステムが固定パイプラインなら、まずマルチホップ質問の割合を計測してみてほしい。30%を超えるなら、Agentic RAGへの移行を検討する価値がある。

---

## 参考文献

1. "A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces" (2026) [arXiv:2602.03442](https://arxiv.org/abs/2602.03442)
2. "Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers" (2025) [arXiv:2506.00054](https://arxiv.org/abs/2506.00054)
3. "Ragas: Automated Evaluation of Retrieval Augmented Generation" (2023) [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
