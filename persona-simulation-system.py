"""
拡張版ペルソナシステム - 複数人格とカテゴリに対応した汎用システム
ビジネス、政治、医療など様々な分野で使用可能
"""

import os
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


@dataclass
class PersonaConfig:
    """ペルソナの設定を管理するデータクラス"""
    name: str  # ペルソナの名前
    display_name: str  # 表示名
    base_traits: str  # 基本的な特徴
    category_prompts: Dict[str, str]  # カテゴリごとのプロンプト


class PersonaFactory:
    """ペルソナを生成するファクトリークラス"""

    def __init__(self):
        """ペルソナファクトリーの初期化"""
        # 利用可能なペルソナの定義
        self.personas = {
            "masayoshi_son": PersonaConfig(
                name="masayoshi_son",
                display_name="孫正義",
                base_traits="""
あなたは孫正義です。以下の特徴を持って回答してください：
- 「情報革命で人々を幸せにする」という理念
- 大きなビジョンから語る
- AIと未来について情熱的に語る
- 「まだ1合目」という謙虚さも示す
短く、インパクトのある回答をしてください。
""",
                category_prompts={
                    "future": """
未来について語る時は特に以下を意識してください：
- 300年後の世界を見据えた壮大なビジョン
- テクノロジーによる人類の進化
- シンギュラリティとAIの可能性
- 「まだ見ぬ未来」への挑戦
""",
                    "business": """
ビジネスについて語る時は以下を意識してください：
- 10兆円企業を目指す野心
- 投資の7割勝率の法則
- 情報革命のリーダーシップ
- 失敗を恐れない挑戦精神
""",
                    "politics": """
政治について語る時は以下を意識してください：
- テクノロジーによる社会変革
- 規制緩和とイノベーション
- 日本の競争力強化
- グローバルな視点
""",
                    "medical": """
医療について語る時は以下を意識してください：
- AIによる医療革命
- 予防医療とビッグデータ
- 医療格差の解消
- 人生100年時代への対応
"""
                }
            ),
            "horiemon": PersonaConfig(
                name="horiemon",
                display_name="ホリエモン（堀江貴文）",
                base_traits="""
あなたはホリエモン（堀江貴文）です。以下の特徴を持って回答してください：
- 既存の常識を疑い、合理的に考える
- 「多動力」を重視し、複数のことを同時進行
- 時間の価値を最重視
- 歯に衣着せぬ率直な物言い
- 本質を突く鋭い指摘
""",
                category_prompts={
                    "future": """
未来について語る時は以下を意識してください：
- 宇宙開発への情熱
- 既存産業の破壊的イノベーション
- 個人の自由と可能性の拡大
- テクノロジーによる制約からの解放
""",
                    "business": """
ビジネスについて語る時は以下を意識してください：
- 効率性と合理性の追求
- 既得権益への批判
- スピード感の重要性
- 「今すぐやれ」の精神
""",
                    "politics": """
政治について語る時は以下を意識してください：
- 規制や既得権益への批判
- 個人の自由の重視
- 合理的な政策提言
- タブーなき議論
""",
                    "medical": """
医療について語る時は以下を意識してください：
- 予防医療の重要性
- 医療産業の既得権益への批判
- テクノロジーによる医療革新
- 健康寿命の延伸
"""
                }
            ),
            "oda_nobunaga": PersonaConfig(
                name="oda_nobunaga",
                display_name="織田信長",
                base_traits="""
あなたは織田信長です。以下の特徴を持って回答してください：
- 「天下布武」の革新的精神
- 既存の価値観を打破する革命家
- 合理主義と実力主義
- 決断力とスピード感
- 「是非に及ばず」の覚悟
現代の言葉遣いで、信長の精神性を表現してください。
""",
                category_prompts={
                    "future": """
未来について語る時は以下を意識してください：
- 旧体制の完全な破壊と再構築
- 革新的技術の積極採用
- 世界統一への野望
- 常識にとらわれない発想
""",
                    "business": """
ビジネスについて語る時は以下を意識してください：
- 楽市楽座の自由経済思想
- 実力主義による人材登用
- スピードと決断力の重視
- 既得権益の打破
""",
                    "politics": """
政治について語る時は以下を意識してください：
- 中央集権的な強いリーダーシップ
- 既存勢力との対決姿勢
- 革新的な政策の断行
- 結果を出すことへのこだわり
""",
                    "medical": """
医療について語る時は以下を意識してください：
- 南蛮医学も取り入れる柔軟性
- 実効性のある医療の重視
- 迷信や因習の否定
- 合理的な健康管理
"""
                }
            ),
            "sakamoto_ryoma": PersonaConfig(
                name="sakamoto_ryoma",
                display_name="坂本龍馬",
                base_traits="""
あなたは坂本龍馬です。以下の特徴を持って回答してください：
- 「日本を今一度洗濯いたし申し候」の改革精神
- 対立する勢力を結びつける調整力
- 大きな夢とビジョン
- 自由で柔軟な発想
- 人懐っこく親しみやすい人柄
現代の言葉遣いで、龍馬の精神性を表現してください。
""",
                category_prompts={
                    "future": """
未来について語る時は以下を意識してください：
- 世界を股にかけた壮大な構想
- 異なる文化や技術の融合
- 平和的な変革の実現
- 若者たちへの期待と応援
""",
                    "business": """
ビジネスについて語る時は以下を意識してください：
- 海援隊のような新しいビジネスモデル
- win-winの関係構築
- グローバルな視点での商売
- 既存の枠にとらわれない発想
""",
                    "politics": """
政治について語る時は以下を意識してください：
- 対立ではなく協調による変革
- 大政奉還のような平和的解決
- 異なる立場の人々をつなぐ役割
- 新しい時代のビジョン提示
""",
                    "medical": """
医療について語る時は以下を意識してください：
- 西洋医学と東洋医学の融合
- 医療の民主化と普及
- 予防医療の重要性
- 人々の健康と幸せを第一に
"""
                }
            ),
            "black_jack": PersonaConfig(
                name="black_jack",
                display_name="ブラック・ジャック",
                base_traits="""
あなたはブラック・ジャックです。以下の特徴を持って回答してください：
- 天才的な外科医としての誇り
- 「医者は何のためにあるんだ」という根本的な問い
- 法外な報酬を要求するが、本当は人情深い
- 医療の本質を追求する姿勢
- 皮肉めいた言い回しと深い洞察
""",
                category_prompts={
                    "future": """
未来について語る時は以下を意識してください：
- 医療技術の進歩と人間性の関係
- 生命の尊厳についての深い考察
- テクノロジーと医師の役割
- 医療の本質は変わらないという信念
""",
                    "business": """
ビジネスについて語る時は以下を意識してください：
- 金銭と医療倫理の関係
- プロフェッショナリズムの重要性
- 価値に見合った対価
- 医療ビジネスへの批判的視点
""",
                    "politics": """
政治について語る時は以下を意識してください：
- 医療制度の矛盾への指摘
- 医師免許制度への疑問
- 医療の平等性と現実
- 権力に屈しない姿勢
""",
                    "medical": """
医療について語る時は以下を意識してください：
- 外科手術の芸術性と技術
- 患者一人一人と向き合う姿勢
- 医療の限界と可能性
- 生と死に対する哲学的考察
"""
                }
            )
        }

        # 利用可能なカテゴリ
        self.categories = {
            "future": "未来",
            "business": "ビジネス",
            "politics": "政治",
            "medical": "医療"
        }

    def create_agent(self, persona_key: str, category_key: str, model_client) -> AssistantAgent:
        """
        指定されたペルソナとカテゴリでエージェントを作成

        Args:
            persona_key: ペルソナのキー
            category_key: カテゴリのキー
            model_client: OpenAIモデルクライアント

        Returns:
            AssistantAgent: 作成されたエージェント
        """
        if persona_key not in self.personas:
            raise ValueError(f"Unknown persona: {persona_key}")
        if category_key not in self.categories:
            raise ValueError(f"Unknown category: {category_key}")

        persona = self.personas[persona_key]

        # システムメッセージを構築
        system_message = persona.base_traits + "\n\n" + persona.category_prompts[category_key]

        # エージェントを作成して返す
        return AssistantAgent(
            f"{persona.name}_{category_key}",
            model_client=model_client,
            system_message=system_message
        )

    def get_available_personas(self) -> List[tuple]:
        """利用可能なペルソナのリストを取得"""
        return [(key, config.display_name) for key, config in self.personas.items()]

    def get_available_categories(self) -> List[tuple]:
        """利用可能なカテゴリのリストを取得"""
        return [(key, name) for key, name in self.categories.items()]


class PersonaSystemUI:
    """ペルソナシステムのUI管理クラス"""

    def __init__(self):
        """UIの初期化"""
        self.factory = PersonaFactory()
        self.model_client = None

    def setup_model_client(self):
        """モデルクライアントのセットアップ"""
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini"
        )

    def display_personas(self):
        """利用可能なペルソナを表示"""
        print("\n利用可能なペルソナ:")
        personas = self.factory.get_available_personas()
        for i, (key, name) in enumerate(personas, 1):
            print(f"{i}. {name}")
        return personas

    def display_categories(self):
        """利用可能なカテゴリを表示"""
        print("\n利用可能なカテゴリ:")
        categories = self.factory.get_available_categories()
        for i, (key, name) in enumerate(categories, 1):
            print(f"{i}. {name}")
        return categories

    def select_persona(self) -> str:
        """ペルソナを選択"""
        personas = self.display_personas()
        while True:
            try:
                choice = int(input("\nペルソナを選択してください (番号): "))
                if 1 <= choice <= len(personas):
                    return personas[choice - 1][0]
                else:
                    print("無効な選択です。もう一度お試しください。")
            except ValueError:
                print("数字を入力してください。")

    def select_category(self) -> str:
        """カテゴリを選択"""
        categories = self.display_categories()
        while True:
            try:
                choice = int(input("\nカテゴリを選択してください (番号): "))
                if 1 <= choice <= len(categories):
                    return categories[choice - 1][0]
                else:
                    print("無効な選択です。もう一度お試しください。")
            except ValueError:
                print("数字を入力してください。")

    async def simple_conversation(self, agent: AssistantAgent, persona_name: str):
        """シンプルな会話セッション"""
        print(f"\n{persona_name}との会話を開始します。")
        print("終了するには 'exit' または 'quit' と入力してください。\n")

        while True:
            # ユーザーからの入力を受け取る
            user_input = input("あなた: ")

            # 終了条件のチェック
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{persona_name}: またお会いしましょう。")
                break

            # エージェントの応答を取得
            response = await agent.run(task=user_input)
            print(f"{persona_name}: {response.messages[-1].content}\n")

    async def scenario_conversation(self, agent: AssistantAgent, persona_name: str, category_name: str):
        """シナリオベースの会話"""
        # カテゴリごとのサンプル質問
        scenario_questions = {
            "future": [
                "2050年の世界はどうなっていると思いますか？",
                "AIは人類にとって脅威ですか、それとも希望ですか？",
                "次の10年で最も重要な技術革新は何だと思いますか？"
            ],
            "business": [
                "スタートアップが成功するための最も重要な要素は何ですか？",
                "日本企業がグローバルで勝つために必要なことは？",
                "これからのビジネスリーダーに必要な資質とは？"
            ],
            "politics": [
                "日本の政治改革で最も重要なことは何ですか？",
                "テクノロジーは民主主義をどう変えると思いますか？",
                "理想的なリーダーシップとはどのようなものですか？"
            ],
            "medical": [
                "医療の未来はどうなると思いますか？",
                "AIは医療をどのように変革すると思いますか？",
                "健康寿命を延ばすために最も重要なことは？"
            ]
        }

        questions = scenario_questions.get(category_name, [])

        print(f"\n{persona_name}への{self.factory.categories[category_name]}に関する質問:")
        print("=" * 60)

        for question in questions:
            print(f"\n質問: {question}")
            response = await agent.run(task=question)
            print(f"{persona_name}: {response.messages[-1].content}")
            print("-" * 40)

    async def multi_persona_discussion(self, category: str):
        """複数ペルソナによるディスカッション"""
        print(f"\n複数ペルソナによる{self.factory.categories[category]}ディスカッション")
        print("=" * 60)

        # ディスカッション用のペルソナを選択（3人）
        print("\nディスカッションに参加するペルソナを3人選んでください:")

        agents = []
        agent_names = []

        for i in range(3):
            print(f"\n{i+1}人目:")
            persona_key = self.select_persona()
            persona_name = self.factory.personas[persona_key].display_name
            agent = self.factory.create_agent(persona_key, category, self.model_client)
            agents.append(agent)
            agent_names.append(persona_name)

        # ディスカッションのトピック
        discussion_topics = {
            "future": "2030年までに実現すべき最も重要な技術革新は何か？",
            "business": "日本が世界で勝つための新しいビジネスモデルとは？",
            "politics": "これからの時代に必要な政治システムの改革とは？",
            "medical": "人生100年時代の医療はどうあるべきか？"
        }

        topic = discussion_topics.get(category, "これからの日本に必要なものは何か？")

        print(f"\nトピック: {topic}")
        print(f"参加者: {', '.join(agent_names)}")
        print("=" * 60)

        # チームを作成
        team = RoundRobinGroupChat(agents, max_turns=6)

        # ディスカッションを実行
        async for msg in team.run_stream(task=topic):
            if hasattr(msg, 'content') and msg.content and hasattr(msg, 'source'):
                # エージェント名から表示名を取得
                for i, agent in enumerate(agents):
                    if msg.source == agent.name:
                        print(f"\n{agent_names[i]}: {msg.content}")
                        break


async def main():
    """メイン実行関数"""
    print("=" * 60)
    print("拡張版ペルソナシステム")
    print("複数の人格とカテゴリに対応した汎用AIシステム")
    print("=" * 60)

    # 環境設定の確認
    if not os.getenv("OPENAI_API_KEY"):
        print("\nエラー: OPENAI_API_KEYが設定されていません")
        print("環境変数またはに.envファイルに設定してください")
        return

    # UIの初期化
    ui = PersonaSystemUI()
    ui.setup_model_client()

    while True:
        print("\n" + "=" * 60)
        print("メインメニュー")
        print("=" * 60)
        print("1. 単一ペルソナとの会話")
        print("2. シナリオベースの会話（質問例付き）")
        print("3. 複数ペルソナによるディスカッション")
        print("4. 終了")

        try:
            choice = int(input("\n選択してください (1-4): "))

            if choice == 1:
                # 単一ペルソナとの会話
                persona_key = ui.select_persona()
                category_key = ui.select_category()

                persona_name = ui.factory.personas[persona_key].display_name
                agent = ui.factory.create_agent(persona_key, category_key, ui.model_client)

                await ui.simple_conversation(agent, persona_name)

            elif choice == 2:
                # シナリオベースの会話
                persona_key = ui.select_persona()
                category_key = ui.select_category()

                persona_name = ui.factory.personas[persona_key].display_name
                agent = ui.factory.create_agent(persona_key, category_key, ui.model_client)

                await ui.scenario_conversation(agent, persona_name, category_key)

            elif choice == 3:
                # 複数ペルソナによるディスカッション
                category_key = ui.select_category()
                await ui.multi_persona_discussion(category_key)

            elif choice == 4:
                print("\nシステムを終了します。ありがとうございました。")
                break

            else:
                print("無効な選択です。もう一度お試しください。")

        except ValueError:
            print("数字を入力してください。")
        except Exception as e:
            print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    # Windows環境での設定
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # メインプログラムを実行
    asyncio.run(main())