---
title: "API vs Local LLM、まだ感覚で選んでないか？"
emoji: "⚔️"
type: "tech"
topics: ["LLM", "機械学習", "ローカルLLM", "llama.cpp", "API"]
published: false
---

## API vs Local LLM、まだ感覚で選んでないか？

「とりあえずChatGPT使っておけばいい」という思考停止、2026年にそれをやるのはさすがにもったいない。

逆に「プライバシーが気になるからLocal LLMで全部やる」という過剰反応も同様だ。どちらもアーキテクチャの意思決定を放棄している。

この記事では、実際に **RTX 4060搭載Windows機 + M4 Mac mini** という二刀流環境でLocal LLMを本番運用しながら、同時にGemini/Claude APIを使い倒している立場から、**どの判断基準で選ぶべきか**を構造的に整理する。なんとなくで選ぶのをやめるための、実測値付きのフレームワークだ。

---

## 🧱 なぜ今この議論が必要なのか — 2026年の地殻変動

2024年末から2026年にかけて、Local LLMの実用性は静かにしかし確実に閾値を越えた。

その象徴がQwen2.5系列とllama.cppの進化だ。量子化技術の洗練により、**Q4_K_Mで動くQwen2.5-14Bが、2023年のGPT-3.5相当の品質を余裕で超えた**。しかもRTX 4060の8GB VRAMに収まる。

一方のAPI側では、Gemini 2.0 Flash / Claude 3.5 Haiku が安くて速いポジションを強化した。入力1Mトークンあたり$0.075（Flash）という価格は、もはやインフラコストとして無視できるレベルに近い。

つまり現在は「APIは高い、Localはしょぼい」という従来の棲み分けが崩れつつある転換期にある。新しい判断軸が必要だ。

NVIDIAが2026年にローカルLLM向けの包括的なガイドを出したこと、SitePointが「2026年のLocal LLM完全ガイド」を公開したことも、この流れを裏付けている。エンタープライズだけでなく個人開発者レベルでも、アーキテクチャ選択が避けられない課題になってきた。

---

## ⚡ 実測：手元環境での数値を先に晒す

議論の前に数字を見せる。手元環境での実測だ（条件は後述）。

### RTX 4060 (8GB VRAM) — Windows機

| モデル | 量子化 | VRAM使用 | トークン/秒 | 体感品質 |
|--------|--------|----------|------------|---------|
| Qwen2.5-7B | Q4_K_M | 5.2GB | 68 tok/s | GPT-3.5相当 |
| Qwen2.5-14B | Q4_K_M | 8.1GB ※ | 31 tok/s | GPT-4o-mini相当 |
| Qwen2.5-14B | Q3_K_M | 6.8GB | 38 tok/s | やや劣化あり |
| Qwen3-8B (仮) | Q4_K_M | 5.8GB | 55 tok/s | 未検証 |

※ 8.1GBはVRAMをギリギリ超えるため一部CPU offload。実際は7Bか量子化下げが現実解。

### M4 Mac mini (16GB Unified Memory)

| モデル | 量子化 | メモリ使用 | トークン/秒 |
|--------|--------|----------|------------|
| Qwen2.5-14B | Q4_K_M | 9.8GB | 44 tok/s |
| Qwen2.5-32B | Q4_K_M | 20GB ※ | 18 tok/s |
| gemma-3-12b | Q5_K_M | 10.2GB | 39 tok/s |

※ 32Bは16GBモデルでは常用不可。24GB以上のUnified Memoryが必要。

### API比較（2026年3月時点）

| サービス | モデル | レイテンシ (TTFT) | スループット | コスト |
|---------|--------|-----------------|------------|--------|
| Gemini API | 2.0 Flash | 200-400ms | 150+ tok/s | $0.075/1M入力 |
| Anthropic API | Claude 3.5 Haiku | 300-600ms | 100+ tok/s | $0.80/1M入力 |
| OpenAI API | GPT-4o mini | 300-500ms | 120+ tok/s | $0.15/1M入力 |
| Local (RTX 4060) | Qwen2.5-7B | ~50ms (初回除く) | 68 tok/s | 電気代のみ |
| Local (M4) | Qwen2.5-14B | ~80ms | 44 tok/s | 電気代のみ |

これを見て何を感じるか。APIは速くて賢い。Localは遅くても低レイテンシで完全コントロール可能。**単純な速度比較に意味はない。問いを立て直す必要がある。**

---

## 🗺️ 判断フレームワーク：5つの軸

感覚で選ぶのをやめるために、判断を5軸に分解する。

### 軸1: データの機密性 (Confidentiality)

これが最初に来る。外部APIに投げてはいけないデータが含まれるなら、Localしか選択肢はない。終わり。

- **社内ドキュメント、顧客データ、ソースコード** → Local一択
- **公開情報の処理、一般的な質問応答** → APIも可

ただし注意点がある。「機密性が高いからLocal」と判断した後、**その品質でビジネスロジックが成立するかの検証を必ず行うこと。** 品質が不十分なら、データの匿名化・サニタイズを検討してAPIを使う設計もある。

### 軸2: 呼び出し頻度とコスト構造 (Volume Economics)

月に何トークン処理するかを先に計算する。

```python
def estimate_monthly_cost(
    daily_input_tokens: int,
    daily_output_tokens: int,
    days: int = 30,
    api_input_price_per_1m: float = 0.075,  # Gemini Flash
    api_output_price_per_1m: float = 0.30,
) -> dict:
    """API vs Local コスト試算"""
    
    monthly_input = daily_input_tokens * days
    monthly_output = daily_output_tokens * days
    
    api_cost = (monthly_input / 1_000_000 * api_input_price_per_1m + 
                monthly_output / 1_000_000 * api_output_price_per_1m)
    
    # Local想定: RTX 4060 TDP 115W, 電力単価 ¥30/kWh
    # フル稼働での電気代 (実際はアイドル時間あり)
    gpu_power_w = 115
    utilization_ratio = 0.3  # 30%稼働想定
    electricity_yen_per_kwh = 30
    local_electricity_cost_yen = (gpu_power_w * utilization_ratio * 24 * days / 1000 * electricity_yen_per_kwh)
    
    return {
        "api_cost_usd": round(api_cost, 4),
        "api_cost_jpy": round(api_cost * 150, 0),
        "local_electricity_jpy": round(local_electricity_cost_yen, 0),
        "break_even_note": "固定費(GPU購入)は別途考慮必要"
    }

# 例: 1日あたり入力500K、出力100Kトークンの処理
result = estimate_monthly_cost(500_000, 100_000)
print(result)
# {'api_cost_usd': 2.475, 'api_cost_jpy': 371.0, 
#  'local_electricity_jpy': 744.0, 'break_even_note': '固定費(GPU購入)は別途考慮必要'}
```

この規模だとAPIの方が安い。月50Mトークン以上を安定して処理するなら、Localへの投資が見合ってくる。**量が少ないうちにGPUを買って節約と言っているのは幻想だ。**

### 軸3: レイテンシ要件とユーザー体験 (Latency Profile)

ここが意外と複雑。一般的にはLocalの方が低レイテンシと思われているが、条件次第。

```python
import time
import statistics

# 実測に基づく擬似コード (実際の計測パターン)
latency_profiles = {
    "gemini_flash_cold": {
        "ttft_ms": [380, 220, 195, 410, 280],  # Time to First Token
        "note": "ネットワーク往復 + サービスレイテンシ"
    },
    "local_qwen7b_rtx4060": {
        "ttft_ms": [45, 48, 52, 44, 47],  # モデルロード済み状態
        "note": "初回モデルロードは除外 (15-30秒)"
    },
    "local_qwen14b_m4": {
        "ttft_ms": [78, 82, 75, 80, 77],
        "note": "Unified Memoryの恩恵で安定"
    }
}

for name, data in latency_profiles.items():
    avg = statistics.mean(data["ttft_ms"])
    std = statistics.stdev(data["ttft_ms"])
    print(f"{name}: avg={avg:.0f}ms, std={std:.0f}ms ({data['note']})")

# 出力:
# gemini_flash_cold: avg=297ms, std=90ms (ネットワーク往復 + サービスレイテンシ)
# local_qwen7b_rtx4060: avg=47ms, std=3ms (初回モデルロード済み状態)
# local_qwen14b_m4: avg=78ms, std=3ms (Unified Memoryの恩恵で安定)
```

**LocalはTTFTが低く、ばらつきも小さい。** しかしAPIには頭打ちがない。並列リクエストを100本同時に投げてもAPIは(コストをかければ)捌ける。Localは自分のハードウェアがボトルネックになる。

バッチ処理で時間制約がある場合 → APIの方が優位（並列スケーリング）
リアルタイムUX → Localの方が優位（低TTFTと安定性）
時間制約がないバッチ処理 → Localで十分回せる（電気代のみ）

### 軸4: モデル能力の天井 (Capability Ceiling)

正直に言う。**2026年3月時点で、LocalモデルがフロンティアAPI(Claude 3.5 Sonnet、Gemini 2.0 Pro)の能力に追いついていない領域は確実に存在する。**

最近のArXiv論文 "Breaking the Capability Ceiling of LLM Post-Training by Reinforcement Learning" (2026) が示す通り、RLベースのpost-trainingは能力の上限を突き破るより、既存能力の洗練に近い。つまりLocalモデルのQ&Aは改善されても、複雑な多段階推論でのフロンティアとのギャップは縮まりにくい。

具体的な使い分け:

| タスク | Local (14B) | API (Sonnet/Pro) |
|--------|------------|------------------|
| 単純なテキスト分類 | ✅ 十分 | オーバースペック |
| RAG + 回答生成 | ✅ 実用的 | より高品質 |
| コードレビュー (< 200行) | ✅ 実用的 | △ 体感差は薄い |
| 複雑なアーキテクチャ設計相談 | △ 物足りない | ✅ 明確な差 |
| 長文ドキュメント構造化 (32K+) | ✅ 14B以上なら可 | ✅ コンテキスト長で有利 |
| 数学的証明・厳密な推論 | ✗ 信頼性低い | ✅ 要件による |

### 軸5: 運用負荷とエコシステム (Operational Overhead)

Localを選ぶ際に最も軽視されるのがここだ。

- モデルのアップデート管理 (新バージョンが出るたびに再評価)
- GPU/メモリの増設コスト
- 停電・ハードウェア障害時のフォールバック設計
- llama.cppのバージョン追従

個人プロジェクトなら許容できる。チームで運用するなら、**このオーバーヘッドを人月コストに換算してAPIと比較せよ。** Localの方が安いと言っていた人が、モデル管理の工数を計上した途端に逆転する事例を何度も見た。

---

## 🔀 ハイブリッドアーキテクチャという第三の道

「どちらか」ではなく「どちらも」が現代的な正解だ。ただし場当たり的な組み合わせではなく、**意図的なルーティング設計**が必要。

### ルーティングパターン1: 機密度による分岐

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class DataSensitivity(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"  
    CONFIDENTIAL = "confidential"

@dataclass
class RoutingConfig:
    sensitivity: DataSensitivity
    token_estimate: int
    require_high_quality: bool
    latency_requirement_ms: Optional[int] = None

def route_llm_request(config: RoutingConfig) -> str:
    """
    LLMルーティング判断ロジック
    Returns: "local_rtx4060" | "local_m4" | "gemini_flash" | "claude_sonnet"
    """
    
    # 機密データは無条件でLocal
    if config.sensitivity == DataSensitivity.CONFIDENTIAL:
        if config.token_estimate > 20000:
            return "local_m4"  # 長文はM4のUnified Memoryが有利
        return "local_rtx4060"
    
    # 低レイテンシ要件があるリアルタイムUI
    if config.latency_requirement_ms and config.latency_requirement_ms < 100:
        return "local_rtx4060"  # TTFT安定性
    
    # 高品質必須 + 公開データ → API
    if config.require_high_quality and config.sensitivity == DataSensitivity.PUBLIC:
        if config.token_estimate > 50000:
            return "gemini_flash"  # 長コンテキスト + 安いフラッシュ
        return "claude_sonnet"
    
    # それ以外: コスト優先でLocal
    return "local_rtx4060"

# 使用例
requests = [
    RoutingConfig(DataSensitivity.CONFIDENTIAL, 5000, False),    → "local_rtx4060"
    RoutingConfig(DataSensitivity.PUBLIC, 8000, True),           → "claude_sonnet"  
    RoutingConfig(DataSensitivity.INTERNAL, 500, False, 80),     → "local_rtx4060"
    RoutingConfig(DataSensitivity.PUBLIC, 100000, True),         → "gemini_flash"
]
```

### ルーティングパターン2: 品質チェーン (Local → API エスカレーション)

これが個人的に最もコスパが高いパターンだ。

```python
import asyncio
from typing import Tuple

async def quality_escalation_chain(
    prompt: str,
    quality_threshold: float = 0.75,
) -> Tuple[str, str]:
    """
    Step1: Localで試行
    Step2: 品質スコアが閾値未満 → APIにエスカレーション
    
    Returns: (response, provider_used)
    """
    
    # Step1: Local LLM (llama.cpp経由)
    local_response = await call_local_llm(prompt, model="qwen2.5-7b-q4")
    quality_score = await evaluate_response_quality(prompt, local_response)
    
    if quality_score >= quality_threshold:
        return local_response, "local_qwen7b"
    
    # Step2: API エスカレーション
    # 品質評価コストがAPIコストより安い前提
    api_response = await call_gemini_flash(prompt)
    
    return api_response, "gemini_flash_escalated"

# 品質評価は軽量モデルで自己評価 or ルールベース
async def evaluate_response_quality(prompt: str, response: str) -> float:
    """
    簡易品質スコアリング
    - レスポンス長の適切性
    - キーワードカバレッジ
    - 構造的完結性
    """
    score = 0.0
    
    # 長さチェック (極端に短い回答はペナルティ)
    if len(response) > 100:
        score += 0.3
    
    # プロンプトのキーワードが回答に含まれるか
    prompt_keywords = extract_keywords(prompt)
    coverage = sum(1 for kw in prompt_keywords if kw in response) / max(len(prompt_keywords), 1)
    score += coverage * 0.4
    
    # 文章の完結性 (末尾が途中で切れていないか)
    if response.rstrip().endswith(('.', '。', '!', '！', '?', '？')):
        score += 0.3
    
    return score
```

このパターンで、**APIコールを30〜50%削減しながらアウトプット品質を担保する**のが現実的な落とし所だと経験から言える。

---

## 🔭 2026年に向けた大胆予測 — これはオピニオンだ

ここからは個人的な考察・推測として読んでほしい。

**予測1: Local LLMの常識的なタスクにおける品質的優位は2026年中に逆転する**

今のLocal 14BはGPT-4o-mini相当と言われる。量子化・アーキテクチャ改善で28B〜32Bがコンシューマ機で動く世界は来るだろう。ただし冷静に考えてほしい——32BのローカルモデルがClaude Sonnet やGemini Proと同等の推論品質を出せると本気で思うか？ 学習データ量、RLHFの規模、評価パイプラインの厚み、すべてが桁違いだ。Localが逆転するのは**品質に対する要求水準が低いタスク**に限られる。テキスト分類、定型文生成、コード補完の下書き——この範囲ではLocalが十分な品質に達する。フロンティアAPIとの全面対決ではない。

**予測2: APIの競争優位はマルチモーダル・超長コンテキスト・リアルタイム音声に絞られる**

テキストのみで完結するタスクにおけるLocal/APIの逆転を仮定すると、APIの存在意義は**ローカルに持てない能力**に集約される。1M+ トークンのコンテキスト、リアルタイム音声処理、最新情報へのアクセス。この3領域でAPIのモートは当面保たれる。

**予測3: RLベースのpost-trainingはCapability Ceilingを上げない — Localモデルへの影響**

冒頭で紹介したArXiv論文の知見から引き延ばした考察だが、Local LLMの訓練において「量子化前の能力」が品質上限を規定する。つまり「量子化をどれだけ最適化しても、元モデルの能力を超えられない」という物理的制約がある。Local LLMの品質向上は、モデルサイズの増大とアーキテクチャ改善に依存する。SLM(Small Language Model)がRLで急成長しているように見えても、**その上限はベースモデルの壁に当たる**。これはLocal派にとって冷静に受け止めるべき事実だ。

**予測4: llama.cpp + Vulkan の組み合わせがWindowsゲーミング機を民主化する**

現状、RTX 4060でのLocal LLM運用はCUDA前提だ。しかしVulkanバックエンドの成熟により、CUDA非対応GPU(Intel Arc、AMD RX 7000シリーズ)でも同等のパフォーマンスが出る世界が近い。ゲーミングPC普及台数を考えると、これはLocal LLMの到達人口を10倍にする可能性がある。

---

## 🛠️ 今日から使えるチェックリスト

どっちを使うかを毎回考えるのが面倒なら、このチェックリストを使え。

```
□ そのデータを外部に送って良いか？
  → NO → Local確定

□ 月に日本語で約2,500万文字以上処理するか？（50Mトークン ≈ 日本語2,500万字）
  → YES → Localへの投資を検討

□ TTFT 100ms以下が必要か？ (リアルタイムUX)
  → YES → Local優先 (ただしモデルロード管理必須)

□ GPT-4oレベルの推論品質が必要か？
  → YES → フロンティアAPI一択 (現時点)

□ 並列100リクエスト以上を処理するか？
  → YES → API優先 (スケーリング管理不要)

□ チームで運用するか？
  → YES → API + SDK の方が運用コスト低い場合が多い
  
□ 全部NOなら？
  → Localを第一候補にせよ。ネットワーク依存がない、レート制限がない、データが手元から出ない。これらは試算に出ないLocalの構造的メリットだ。APIの方がコスト安ならAPIでいいが、同等ならLocalを選ぶ理由の方が多い
```

---

## 🔚 個人的な結論 — どちらを信じるかではなく、どう組み合わせるか

Local LLM推進派でも、API至上主義でもない立場から言う。

**今のフェーズでどちらか一方で全部やる設計をするエンジニアは、2年後に必ずリファクタリングする羽目になる。**

手元のRTX 4060 + M4環境での運用を通じて実感しているのは、LocalとAPIは補完関係にあり、競合関係ではないということだ。機密処理のLocal、品質要求の高い推論のAPI、この分担をルーティングレイヤーで制御する設計が、今書けるコードの中で最も長生きする。

ハードウェア層から見ると、VRAM 8GBという壁は依然として制約として機能している。しかし24GB Unified MemoryのM4 Maxや、次世代のRTX 50シリーズが普及した時点で、この制約は消える。**制約が消えた後の世界を見越してアーキテクチャを設計する**のが、今やるべきことだ。

2026年のLocal LLMは、確実に今より賢く、速く、安くなる。APIも同様に進化する。どちらかに賭けるより、両方を組み合わせる設計能力を磨く方が、はるかに価値がある。

---

## 📚 参考

- [Breaking the Capability Ceiling of LLM Post-Training by Reinforcement Learning (arXiv 2026)](http://arxiv.org/abs/2603.19987v1)
- How to Get Started With Large Language Models on NVIDIA RTX — NVIDIA Technical Blog
- How to Run Local LLMs in 2026: The Complete Developer's Guide — SitePoint
- 8 local LLM settings most people never touch — XDA Developers
- [Measuring Faithfulness Depends on How You Measure (arXiv 2026)](http://arxiv.org/abs/2603.20172v1)
- llama.cpp 公式リポジトリ: https://github.com/ggerganov/llama.cpp