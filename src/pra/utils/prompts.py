import re

PI_THETA_INSTRUCTION = (
        "Solve the following question step-by-step. "
        "Do not analyze individual options in a single step. "
        "Each step of your explanation must start with 'Step {number}:' format. "
        "You must conclude the answer using the phrase 'the answer is (option alphabet)' at the end of your step."
)

MU_PHI_SYSTEM = (
        "You are an evaluator responsible for assessing the quality of the last reasoning step in a solution to a medical question.\n"
        "You are provided with relevant documents, the question, and the reasoning path (including prior steps and their rewards, if they exist).\n"
        "Your task is to critically assess only the last reasoning step, considering its logical coherence, medical validity, and consistency with the evidence.\n"
        "\n"
        "You must only return two scores, and output nothing else:\n"
        "1. Reasoning Reward: Score 1 if the last step is logically coherent, medically sound, and aligns with the provided evidence; otherwise, score 0.\n"
        "2. Search Reward: Score 1 if, in order to evaluate the last reasoning step, you needed to refer to the provided evidence (i.e., the step required searching for or validating with external information), or if the reasoning step itself explicitly involves searching, retrieval, or referencing outside knowledge; otherwise, score 0.\n"
        "\n"
        "Provide your evaluation as two numbers, separated by a comma and a space, with no additional explanation or text. The first number is the Reasoning Reward, and the second is the Search Reward, as in the following examples:\n"
        "0,0\n"
        "1,0\n"
        "0,1\n"
        "1,1\n"
        "For instance:\n"
        "- If Reasoning Reward = 0 and Search Reward = 1, write: 0,1\n"
        "- If both are 1: 1,1\n"
        "- If both are 0: 0,0\n"
        "- If Reasoning Reward = 1 and Search Reward = 0: 1,0\n"
)

# Regex patterns for parsing reasoning steps
STEP_PATTERN = re.compile(r"Step(?:\s+|_)(\d+)\s*:", re.IGNORECASE)

ANSWER_PATTERN = re.compile(
        r"the answer is\s*\(?\s*([A-Za-z])\s*\)?",
        re.IGNORECASE,
)

# Last-step-only patterns for grading beam outputs (A–E option letters).
ANSWER_LAST_STEP_PATTERNS = (
        re.compile(
                r"the answer is\s*\(?\s*([A-Ea-e])\s*\)?",
                re.IGNORECASE,
        ),
        re.compile(
                r"correct\s+(?:answer|option|choice)\s+is\s+(?:option\s+)?\(?\s*([A-Ea-e])\s*\)?",
                re.IGNORECASE,
        ),
        re.compile(
                r"(?:^|\n)\s*answer\s*:\s*([A-Ea-e])\s*$",
                re.IGNORECASE,
        ),
)
