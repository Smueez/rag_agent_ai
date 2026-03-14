import os


def read_agent_instruction_file(agent_name: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    instructions_path = os.path.join(current_dir, '..', 'services/agent_services/instructions', f'{agent_name}.txt')
    instructions_path = os.path.abspath(instructions_path)
    lines = ""
    with open(instructions_path, "r") as f:
        tmp = f.readlines()
        lines += "".join(tmp)
    return lines