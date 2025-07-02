"""
孫正義ペルソナシステム - 使用例とテストスクリプト
各種シナリオでの動作確認用
"""

import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()


async def simple_conversation_example():
    """シンプルな会話の例"""
    print("=" * 60)
    print("シンプルな会話の例")
    print("=" * 60)

    # モデルクライアントの設定
    model_client = OpenAIChatCompletionClient(
        model="o4-mini-2025-04-16"
    )

    # 孫正義エージェントの作成
    son_agent = AssistantAgent(
        "masayoshi_son",
        model_client=model_client,
        system_message="""
あなたは孫正義です。以下の特徴を持って回答してください：
- 「情報革命で人々を幸せにする」という理念
- 大きなビジョンから語る
- AIと未来について情熱的に語る
- 「まだ1合目」という謙虚さも示す
短く、インパクトのある回答をしてください。
"""
    )

    # 質問と回答
    questions = [
        "日本のスタートアップについてどう思いますか？",
        "AIの未来について教えてください",
        "若い起業家へのアドバイスをお願いします"
    ]

    for question in questions:
        print(f"\n質問: {question}")
        response = await son_agent.run(task=question)
        print(f"孫正義: {response.messages[-1].content}")


async def investment_evaluation_example():
    """投資評価シナリオの例"""
    print("\n" + "=" * 60)
    print("投資評価シナリオの例")
    print("=" * 60)

    model_client = OpenAIChatCompletionClient(model="o4-mini-2025-04-16")

    # 投資評価に特化した孫正義エージェント
    investment_son = AssistantAgent(
        "investment_son",
        model_client=model_client,
        system_message="""
あなたは投資家としての孫正義です。以下の基準で評価します：
1. この会社は情報革命をリードできるか？
2. 10兆円企業になる可能性は？
3. 創業者の志は高いか？
4. AIをどう活用しているか？
5. 7割以上の勝算はあるか？

評価は厳しく、でも可能性があれば情熱的に語ってください。
"""
    )

    # 架空のスタートアップの評価
    startup_pitch = """
    私たちは「AIドクター」という医療AIサービスを開発しています。
    画像診断の精度は人間の医師を超え、診断時間を90%削減できます。
    すでに10の病院で試験導入され、来年には100病院を目指しています。
    """

    print(f"\nピッチ: {startup_pitch}")
    response = await investment_son.run(task=f"このスタートアップを評価してください: {startup_pitch}")
    print(f"\n孫正義の評価: {response.messages[-1].content}")


async def multi_agent_discussion():
    """マルチエージェントディスカッションの例"""
    print("\n" + "=" * 60)
    print("マルチエージェントディスカッションの例")
    print("=" * 60)

    model_client = OpenAIChatCompletionClient(model="o4-mini-2025-04-16")

    # 孫正義エージェント
    son_agent = AssistantAgent(
        "masayoshi_son",
        model_client=model_client,
        system_message="""
あなたは孫正義です。大きなビジョンを語り、
「情報革命で人々を幸せにする」観点から発言してください。
"""
    )

    # CFOエージェント
    cfo_agent = AssistantAgent(
        "cfo",
        model_client=model_client,
        system_message="""
あなたは慎重なCFOです。財務的なリスクを指摘し、
現実的な数字に基づいた分析を提供してください。
"""
    )

    # CTO エージェント
    cto_agent = AssistantAgent(
        "cto",
        model_client=model_client,
        system_message="""
あなたはCTOです。技術的な実現可能性と
必要なリソースについて具体的に説明してください。
"""
    )

    # チームの作成
    team = RoundRobinGroupChat(
        [son_agent, cfo_agent, cto_agent],
        max_turns=6
    )

    # ディスカッションのトピック
    topic = "1000億円を投資して、日本にAI研究所を作るべきか？"

    print(f"\nトピック: {topic}\n")

    # ストリーミングで会話を表示
    async for msg in team.run_stream(task=topic):
        if hasattr(msg, 'content') and msg.content and hasattr(msg, 'source'):
            print(f"{msg.source}: {msg.content}\n")


def quick_test():
    """クイックテスト - 基本的な動作確認"""
    print("孫正義ペルソナシステム - クイックテスト")
    print("=" * 60)

    # APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEYが設定されていません")
        print("\n以下のいずれかの方法で設定してください：")
        print("\n方法1: .envファイルを作成")
        print("  1. プロジェクトのルートに .env ファイルを作成")
        print("  2. 以下の内容を記述：")
        print("     OPENAI_API_KEY=your-api-key-here")
        print("\n方法2: 環境変数に直接設定")
        print("  Windows: set OPENAI_API_KEY=your-api-key-here")
        print("  Mac/Linux: export OPENAI_API_KEY=your-api-key-here")
        return

    # APIキーの一部を表示（セキュリティのため最初の数文字のみ）
    print(f"✓ APIキーが設定されています (sk-{api_key[3:7]}...)")

    # ライブラリのインポート確認
    try:
        import autogen_agentchat
        print("✓ autogen_agentchat がインストールされています")
    except ImportError:
        print("✗ autogen_agentchat がインストールされていません")
        print("  pip install autogen-agentchat を実行してください")
        return

    try:
        import autogen_ext
        print("✓ autogen_ext がインストールされています")
    except ImportError:
        print("✗ autogen_ext がインストールされていません")
        print("  pip install autogen-ext[openai] を実行してください")
        return

    try:
        import dotenv
        print("✓ python-dotenv がインストールされています")
    except ImportError:
        print("✗ python-dotenv がインストールされていません")
        print("  pip install python-dotenv を実行してください")
        return

    print("\n環境設定が完了しています！")


async def main():
    """メイン実行関数"""
    # クイックテスト
    quick_test()

    # APIキーが設定されていない場合は終了
    if not os.getenv("OPENAI_API_KEY"):
        return

    print("\n実行する例を選択してください：")
    print("1. シンプルな会話")
    print("2. 投資評価シナリオ")
    print("3. マルチエージェントディスカッション")
    print("4. すべて実行")

    choice = input("\n選択 (1-4): ")

    if choice == "1":
        await simple_conversation_example()
    elif choice == "2":
        await investment_evaluation_example()
    elif choice == "3":
        await multi_agent_discussion()
    elif choice == "4":
        await simple_conversation_example()
        await investment_evaluation_example()
        await multi_agent_discussion()
    else:
        print("無効な選択です")


if __name__ == "__main__":
    # Windows環境での設定
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 実行
    asyncio.run(main())