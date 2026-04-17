---
title: "ツール呼び出しでも大きいモデルは勝てなかった"
emoji: "🔧"
type: "tech"
topics: ["LLM", "ローカルLLM", "AI", "Python", "機械学習"]
published: true
---

# ツール呼び出しでも大きいモデルは勝てなかった

LLMにツールを持たせる（function calling / tool use）。これはエージェントの基盤技術であり、RAGの次の進化であり、ローカルLLMの実用性を左右する機能だ。

では、どのモデルがfunction callingで最も正確か。13モデルをQ4_K_M量子化でテストした2026年のベンチマーク（JD Hodges, 2026）の結果は、予想を裏切るものだった。

**97.5%の精度を出したのは3.4GBのモデルだった。25GBのモデルは85%で負けた。**

少なくともこのテスト環境では、大きいモデルが強いという前提は成り立たなかった。

---

## ベンチマーク結果: 13モデル全比較

全モデルQ4_K_M量子化（モデルの精度を維持しつつVRAM使用量を約75%削減する4bit圧縮）、LM Studio経由で統一環境テスト。40ケースのfunction calling精度。サイズはLM Studio上でのロードVRAM（ファイルサイズとは異なる）。

| 順位 | モデル | サイズ | 精度 |
|------|--------|--------|------|
| 1 | **Qwen3.5 4B** | **3.4GB** | **97.5%** |
| 2 | GLM-4.7-Flash | 18GB | 95.0% |
| 2 | Nemotron 3 Nano 4B | 4.2GB | 95.0% |
| 4 | Mistral Nemo 12B | 7.5GB | 92.5% |
| 5 | Qwen3 8B | 5GB | 85.0% |
| 5 | GPT-OSS 20B | 12GB | 85.0% |
| 5 | Nemotron 3 Nano 30B-A3B | 25GB | 85.0% |
| 8 | DeepSeek-R1-Distill 14B | 9GB | 57.5% |
| 8 | Phi-4 Mini | 2.5GB | 57.5% |
| 10 | Gemma 3 4B QAT | 3.2GB | 55.0% |
| 11 | Mistral Small 3.2 24B | 15GB | 42.5% |
| 12 | Hammer 2.1 7B | 4.2GB | 20.0% |
| 13 | xLAM-2 8B FC-R | 4.9GB | 15.0% |

### 3つの衝撃

**衝撃1: 3.4GBが97.5%**
Qwen3.5 4Bは40ケース中39ケースで成功。失敗は1ケースのみ。3.4GBはRTX 4060 8GBのVRAMの半分以下。推論モデルとEmbeddingを同時に動かしても余裕がある。

**衝撃2: サイズだけでは精度を予測できない**

| サイズ | モデル | 精度 |
|-------|--------|------|
| 3.4GB | Qwen3.5 4B | 97.5% (1位) |
| 4.2GB | Nemotron 3 Nano 4B | 95.0% (2位) |
| 7.5GB | Mistral Nemo 12B | 92.5% (4位) |
| 18GB | GLM-4.7-Flash | 95.0% (2位) |
| 25GB | Nemotron 3 Nano 30B-A3B | 85.0% (5位) |

このテスト環境（LM Studio + Q4_K_M）では、モデルサイズだけではfunction calling精度を予測できない。15GBのMistral Smallが42.5%で、3.4GBのQwen3.5に55ポイント差で負ける。ただしn=13かつLM Studioの推論パス依存なので、モデル固有の能力についての一般的結論を出すにはデータが不足している。

**注意: 下位モデルにはLM Studio互換性問題がある**
xLAM-2 8B FC-R（15%）とHammer 2.1（20%）の低スコアは、モデル品質ではなくLM Studioのchat template非互換が原因。xLAM-2は実際にはBerkeley Function Calling Leaderboard（BFCL）で1位のモデルだが、独自のツール呼び出し形式がLM StudioのOpenAI互換APIで正しく変換されなかった。このベンチマークは「LM Studio経由での実用精度」を測っており、モデル固有の能力を測っていない点に注意が必要。

---

## なぜ小さいモデルがfunction callingで勝つのか

### 構造化出力の特殊性

function callingはLLMのタスクの中でも特殊だ。自由文生成ではなく、厳密な構造（JSON、関数名、引数型）に従った出力が必要。

| 要件 | function calling | 自由文生成 |
|------|-----------------|-----------|
| 出力形式 | JSON（厳密なスキーマ準拠） | 自然言語（柔軟） |
| 型安全性 | 引数の型が正確必須 | 不要 |
| ハルシネーション許容度 | ゼロ（存在しない関数を呼べない） | ある程度許容 |
| 知識依存度 | 低い（形式遵守が支配的） | 高い（世界知識が品質に直結） |

ポイント: function callingは**知識量よりフォーマット遵守能力**に依存する。大きいモデルの強みは知識量（世界知識、推論能力）にあるが、function callingではそれが活きない。代わりに、学習データの品質とinstruction followingの精度が効く。

Qwen3.5 4Bが強い理由は推測だが:
1. Qwen3.5世代のinstruction tuningがfunction calling形式に最適化されている
2. 4Bパラメータでも構造化出力には十分な表現力がある
3. パラメータ数を絞ることで、学習データの信号がノイズに埋もれにくい

### パラメータ数神話の崩壊（続）

これは8GB VRAMのモデル選択ルール（別記事）で述べた法則の延長だ。

```
法則 (再確認):
  パラメータ数 ≠ タスク性能

追加法則 (function calling):
  構造化出力タスクでは、小さいモデルが大きいモデルに勝つことがある
  理由: 知識量ではなくフォーマット遵守能力が支配的
```

---

## RTX 4060 8GBでのfunction calling実装

ベンチマーク結果を実装に落とし込む。

### 推奨構成

| 目的 | モデル | VRAM | 精度 | 余剰VRAM |
|------|--------|------|------|---------|
| 最高精度 | Qwen3.5 4B Q4_K_M | ~3.4GB | 97.5% | ~4.6GB（Embedding同時起動可） |
| バランス | Nemotron 3 Nano 4B Q4_K_M | ~4.2GB | 95.0% | ~3.8GB |
| マルチターン | Mistral Nemo 12B Q4_K_M | ~7.5GB | 92.5% | ~0.5GB（単独使用推奨） |

### 最小構成コード

```python
# llama.cpp + Qwen3.5-4B で function calling
import subprocess
import json

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search internal documents by query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]

def call_with_tools(user_message: str, tools: list) -> dict:
    """ローカルLLMでfunction callingを実行"""
    # ChatML形式のプロンプト構築
    tools_json = json.dumps(tools, ensure_ascii=False)
    prompt = f"""<|im_start|>system
You are a helpful assistant with access to tools.
Available tools: {tools_json}
When you need to use a tool, respond with JSON:
{{"name": "function_name", "arguments": {{"key": "value"}}}}
Only call a tool if the user's request requires it.
<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""

    result = subprocess.run(
        ["llama-cli", "-m", "qwen3.5-4b-q4_k_m.gguf",
         "-p", prompt, "-n", "200", "--temp", "0.1",
         "--grammar-file", "json.gbnf"],  # JSON文法制約
        capture_output=True, text=True
    )

    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"error": "Failed to parse", "raw": result.stdout}

# 実行例
result = call_with_tools(
    "東京の天気を教えて",
    TOOLS
)
# 期待出力: {"name": "get_weather", "arguments": {"location": "Tokyo"}}
```

### GBNF文法によるJSON保証

llama.cppのGBNF（Generalized Backus-Naur Form）文法制約を使うと、LLMの出力をJSON形式に強制できる。これにより構造化出力の信頼性がさらに上がる。

```
# json.gbnf (基本的なJSON文法)
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? ws "}"
array  ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" [^"\\]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
```

GBNF制約なしの場合、モデルが自由にテキストを混ぜてパース失敗が頻発する:

```
# GBNF なし — モデルが余計なテキストを混ぜる
{"name": "get_weather", "arguments": {"location": "Tokyo"}} Let me check the weather for you...
```

GBNF制約ありの場合、各トークンが文法規則に準拠するよう制約されるため、構文的に不正なJSONは生成されにくい。ただしトークン上限に達した場合は不完全なJSONになる可能性があり、意味的な正確性（正しい関数名や引数値）は保証されない。推論速度に若干のオーバーヘッドが発生する。

---

## function calling × Agentic RAG の統合

function callingが使えると、前回のAgentic RAG記事（別記事）で紹介したエージェント型RAGがさらに安定する。

```python
# function calling + Agentic RAG の統合実装
ARAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "キーワードベースでドキュメント検索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "意味ベクトルでドキュメント検索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "ユーザーに最終回答を提供",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    }
]

def agentic_rag_loop(question: str, max_steps: int = 3) -> str:
    """function calling ベースの Agentic RAG"""
    context = []

    for step in range(max_steps):
        # 現在のコンテキストでfunction callを要求
        tool_call = call_with_tools(
            f"Question: {question}\n"
            f"Retrieved so far: {json.dumps(context[:3])}\n"
            f"Call a search tool or answer directly.",
            ARAG_TOOLS
        )

        if tool_call.get("name") == "answer":
            return tool_call["arguments"]["text"]

        elif tool_call.get("name") == "keyword_search":
            results = keyword_search(tool_call["arguments"]["query"])
            context.extend(results)

        elif tool_call.get("name") == "semantic_search":
            results = semantic_search(tool_call["arguments"]["query"])
            context.extend(results)

    # max_stepsに達したら強制回答
    return call_with_tools(
        f"Based on: {context}\nAnswer: {question}",
        [ARAG_TOOLS[2]]  # answerツールのみ
    )["arguments"]["text"]
```

Qwen3.5 4B（97.5%精度）でこのループを回せば、function callingの失敗でエージェントが暴走するリスクを最小化できる。しかも3.4GBなので、BGE-M3 Embedding（1.5GB）と同時に起動しても5GB以内。RTX 4060 8GBに余裕で収まる。

---

## 避けるべきモデルと選ぶべきモデル

### 避けるべきモデル

**LM Studio環境で精度が出なかったモデル:**

| モデル | サイズ | 精度 | 理由 |
|--------|--------|------|------|
| Mistral Small 3.2 24B | 15GB | 42.5% | VRAMに見合わない精度 |
| DeepSeek-R1-Distill 14B | 9GB | 57.5% | 推論特化、構造化出力は苦手 |

※ xLAM-2 8B FC-R（15%）とHammer 2.1（20%）はLM Studioのchat template非互換が原因の可能性が高いため、モデル品質の評価としては参考にならない。

### 用途別推奨

| 用途 | 推奨モデル | サイズ | 精度 | 理由 |
|------|----------|--------|------|------|
| Agentic RAG | Qwen3.5 4B | 3.4GB | 97.5% | 最高精度 + Embedding同時起動可 |
| マルチターンAgent | Mistral Nemo 12B | 7.5GB | 92.5% | シーケンシャル処理に強い |
| コスト最小 | Nemotron Nano 4B | 4.2GB | 95.0% | 高精度 + 省VRAM |
| 速度最優先 | GLM-4.7-Flash | 18GB | 95.0% | 52 t/s (ただしVRAM大) |
| 8GBで知識+ツール | Qwen3.5 4B + 32B切替 | 可変 | 可変 | ツール呼び出しは4B、知識回答は32B |

最後のパターンが興味深い。ツール呼び出し判定には3.4GBの小さいモデル、実際の知識回答には32Bの大きいモデル、という2段構成。ツール選択の精度と知識の深さを両立できる。

---

## function callingはローカルLLMの次のフロンティア

この記事の要点:

1. **function calling精度はモデルサイズだけでは予測できない**: 3.4GBのQwen3.5 4Bが25GBモデルを上回った（LM Studio環境）
2. **構造化出力は知識量より形式遵守能力に依存する傾向がある**: 大きいモデルの強み（知識量）が活きにくいタスク
3. **RTX 4060 8GBはfunction callingの好環境**: Top3モデルがすべて8GB以内に収まる
4. **GBNF文法制約でJSON構文エラーを大幅に削減可能**: llama.cppの強力機能
5. **ベンチマーク結果は推論サーバー依存**: LM Studioの互換性が下位モデルのスコアに影響している

ローカルLLMの価値は「大きいモデルを動かすこと」から「適切なモデルを適切なタスクに使うこと」に移行している。function callingはその象徴的な例だ。ただし、推論サーバーとの互換性が結果を大きく左右するため、自分の環境で検証することが前提になる。

---

## 参考文献

1. JD Hodges. "I Tested 13 Local LLMs on Tool Calling | 2026 Eval Results" (2026)
2. "Small Language Models for Efficient Agentic Tool Calling" (2025) [arXiv:2512.15943](https://arxiv.org/abs/2512.15943)
3. "TinyAgent: Function Calling at the Edge" (2024) [arXiv:2409.00608](https://arxiv.org/abs/2409.00608)
4. "ToolACE: Winning the Points of LLM Function Calling" (2024) [arXiv:2409.00920](https://arxiv.org/abs/2409.00920)
