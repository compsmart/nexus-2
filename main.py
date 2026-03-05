"""NEXUS-2 Interactive CLI.

Usage:
    python main.py                   # Start chat loop
    python main.py --no-llm          # Start without loading LLM (memory-only mode)
    python main.py --device cpu      # Force CPU

Commands in chat:
    /quit       - Exit
    /memory     - Show memory stats
    /train      - Trigger training (opens train.py)
    /benchmark  - Run benchmarks
    /tools      - List available tools
    /clear      - Clear conversation history
"""

import argparse
import json
import sys

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init()
except ImportError:
    class _NoColor:
        def __getattr__(self, _):
            return ""
    Fore = _NoColor()
    Style = _NoColor()

from nexus2.agent import Nexus2Agent
from nexus2.config import NexusConfig


def print_banner():
    print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  NEXUS-2 :: Neural Memory Agent{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Neural encoder + N-hop reasoning + Qwen2.5-7B{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
    print(f"  Commands: /quit  /memory  /tools  /clear")
    print()


def handle_command(agent: Nexus2Agent, cmd: str) -> bool:
    """Handle slash commands. Returns True if should continue, False to exit."""
    cmd = cmd.strip().lower()

    if cmd == "/quit":
        return False

    elif cmd == "/memory":
        stats = agent.memory.get_stats()
        print(f"\n{Fore.YELLOW}Memory Statistics:{Style.RESET_ALL}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()

    elif cmd == "/tools":
        tools = agent.tools.list_tools()
        print(f"\n{Fore.YELLOW}Available Tools ({len(tools)}):{Style.RESET_ALL}")
        for t in tools:
            print(f"  - {t}")
        print()

    elif cmd == "/clear":
        agent._messages.clear()
        agent._prev_user_text = None
        print(f"{Fore.YELLOW}Conversation cleared.{Style.RESET_ALL}\n")

    elif cmd == "/train":
        print(f"{Fore.YELLOW}Run: python train.py --phase all{Style.RESET_ALL}\n")

    elif cmd == "/benchmark":
        print(f"{Fore.YELLOW}Run: python benchmark.py --suite all{Style.RESET_ALL}\n")

    else:
        print(f"{Fore.RED}Unknown command: {cmd}{Style.RESET_ALL}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="NEXUS-2 Interactive Agent")
    parser.add_argument("--no-llm", action="store_true", help="Start without LLM")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cpu")
    parser.add_argument("--message", type=str, help="Non-interactive mode: send a single message and exit")
    args = parser.parse_args()

    # In non-interactive mode redirect stdout to stderr so banner text doesn't
    # pollute the JSON response the UI parses.
    _real_stdout = sys.stdout
    if args.message:
        sys.stdout = sys.stderr

    print_banner()

    config = NexusConfig()
    if args.device != "auto":
        config.device = args.device

    print(f"  Loading agent...", flush=True)
    agent = Nexus2Agent(
        config=config,
        device=args.device,
        load_llm=not args.no_llm,
        load_checkpoints=True,
    )
    agent.start()

    # --- Non-interactive mode (for UI) --------------------------------
    if args.message:
        try:
            response = agent.interact(args.message)
        except Exception as e:
            agent.stop()
            sys.stdout = _real_stdout
            print(json.dumps({"response": "", "error": str(e)}))
            return
        finally:
            agent.stop()
        sys.stdout = _real_stdout
        print(json.dumps({"response": response}))
        return

    stats = agent.get_stats()
    print(f"  Memory: {stats['memory']['size']} entries")
    print(f"  Encoder: {stats['memory']['encoder']}")
    print(f"  LLM: {'loaded' if stats['llm_loaded'] else 'not loaded'}")
    print(f"  Device: {stats['device']}")
    print()

    try:
        while True:
            try:
                user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not handle_command(agent, user_input):
                    break
                continue

            try:
                response = agent.interact(user_input)
                print(f"\n{Fore.BLUE}NEXUS-2:{Style.RESET_ALL} {response}\n")
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}(interrupted){Style.RESET_ALL}\n")
            except Exception as e:
                print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}\n")

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
        agent.stop()
        print(f"{Fore.GREEN}Goodbye.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
