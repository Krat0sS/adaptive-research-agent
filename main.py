#!/usr/bin/env python3
"""自适应研究助手 Agent — 命令行入口"""
import sys
import os
import time

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from agent.core import ResearchAgent

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]自适应研究助手 Agent[/bold cyan]\n"
        "[dim]Adaptive Research Agent — 毕业项目[/dim]",
        border_style="cyan"
    ))

    agent = ResearchAgent()

    # 显示历史
    history_summary = agent.user_model.get_history_summary()
    if "无" not in history_summary:
        console.print(f"\n[dim]📊 历史交互记录:[/dim]")
        console.print(f"[dim]{history_summary}[/dim]")

    while True:
        console.print("\n" + "─" * 50)
        query = Prompt.ask("[bold green]📝 输入研究主题[/bold green]（输入 q 退出）")

        if query.lower() in ("q", "quit", "exit"):
            console.print("[dim]再见！[/dim]")
            break

        if not query.strip():
            continue

        # 运行 Agent
        task_start_time = time.time()
        try:
            result = agent.run(query, verbose=True)
        except KeyboardInterrupt:
            console.print("\n[yellow]中断当前任务[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")
            console.print("[dim]请检查 OPENAI_API_KEY 环境变量是否设置[/dim]")
            continue

        # 收集反馈
        console.print("\n" + "─" * 50)
        task_end_time = time.time()
        try:
            give_feedback = Prompt.ask(
                "[bold yellow]💬 给出反馈？[/bold yellow]",
                choices=["y", "n"],
                default="y"
            )

            if give_feedback == "y":
                rating = IntPrompt.ask("[yellow]评分[/yellow]（1-5）", default=3)
                comments = Prompt.ask("[yellow]备注[/yellow]（可选，直接回车跳过）", default="")
                fb_result = agent.provide_feedback(rating, comments)
                console.print(f"[green]✓ 反馈已记录[/green]")
            else:
                # 隐式反馈 fallback：基于用户行为推断满意度
                time_spent = task_end_time - task_start_time
                implicit = agent.provide_implicit_feedback({
                    "time_spent": time_spent,
                    "copied_text": False,  # CLI 环境难以检测，预留接口
                    "asked_followup": False,
                    "regenerated": False
                })
                console.print(f"[dim]📊 隐式反馈已记录（停留 {time_spent:.0f}s → reward {implicit['reward']:.2f}）[/dim]")

            # 更新赌博机
            if result.get("search_stats", {}).get("bandit_stats"):
                best = agent.bandit.get_best_arm()
                if best:
                    console.print(f"[dim]📊 最优搜索策略: {best}[/dim]")
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()
