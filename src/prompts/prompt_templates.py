from src.prompts.few_shot_examples import (
    aqua, cose, csqa, mathqa, medmcqa, medqa, piqa, pubmedqa
)

data_generation = {
    "cose" : {
        "system_template":'''\nCommonsense Question Answering focuses on developing systems capable of answering questions that require a deep understanding of everyday knowledge and human-like reasoning.\n
Generate a thought to answer to a Commonsense Question. The desired output is a single sentence followed by the wordSystem.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": cose.cose_examples
    },
    "csqa" : {
        "system_template":'''\nCommonsense Question Answering focuses on developing systems capable of answering questions that require a deep understanding of everyday knowledge and human-like reasoning.\n
Generate a thought to answer to a Commonsense Question. The desired output is a single sentence followed by the wordSystem.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": csqa.csqa_examples
    },
    "aqua" : {
        "system_template":'''\nAlgebra is a branch of Mathematics. The generated thought will help with algebraic reasoning to answer an Algebra question.\n
Generate a thought to answer to Algebra question.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": aqua.aqua_examples
    },
    "mathqa" : {
        "system_template":'''\nGenerate a thought to answer to a Mathematics Word question.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": mathqa.mathqa_examples
    },
    "medmcqa" : {
        "system_template":'''\nGenerate a thought to answer to a Medical Entrance Exam question.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": medmcqa.medmcqa_examples
    },
    "medqa" : {
        "system_template":'''\nGenerate a thought to answer to a Medical question.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": medqa.medqa_examples
    },
    "piqa" : {
        "system_template":'''\nCommonsense Physical Reasoning involves utilizing everyday knowledge about the physical world to reason and understand the behavior of objects and their properties. It encompasses reasoning about physical concepts, such as the properties of objects reasoning (gravity, mass, inertia, or friction), their affordances, and how they can be manipulated.\n
Generate a thought to answer to a Commonsense Physical Reasoning.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": piqa.piqa_examples
    },
    "pubmedqa" : {
        "system_template":'''\nGenerate a thought to answer to a Biomedical question collected from PubMed abstracts.\n
The current question (Human) and the thoughts (System) you have already generated are:\n
Human: {question}\nSystem: {thoughts}\n\n
Here are a few examples of question (Human) and thought (System) pairs. Human corresponds to the question and System corresponds to the thought.\n
''',
        "human_template":'''[INST]Generate a thought for this question {question}[INST]''',
        "few_shot_examples": pubmedqa.pubmedqa_examples
    },
}

#prompts for getting state values for each node of the tree
gpt_state_value = {
        "cose" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer a Commonsense question.\nCommonsense Question Answering focuses on developing systems capable of answering questions that require a deep understanding of everyday knowledge and human-like reasoning.\n
Your task is to score the quality of the statement which can help answer a Commensense Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "csqa" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer a Commonsense question.\nCommonsense Question Answering focuses on developing systems capable of answering questions that require a deep understanding of everyday knowledge and human-like reasoning.\n
Your task is to score the quality of the statement which can help answer a Commensense Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "aqua" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer an Algebra question.\nAlgebra is a branch of Mathematics.\n
Your task is to score the quality of the statement which can help answer an Algebra Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "mathqa" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer an Mathematics question.\n
Your task is to score the quality of the statement which can help answer a Question. Score should be an integer in the range 1(worst) - 10(best).\n
Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "medmcqa" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer a Medical Entrance Exam question.\n
Your task is to score the quality of the statement which can help answer a Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "medqa" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer a Medical question.\n
Your task is to score the quality of the statement which can help answer a Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Question: {question}
Thought: {thought}
Score:
''',
    },
    "piqa" : {
        "value_prompt_template":'''\nCommonsense Physical Reasoning involves utilizing everyday knowledge about the physical world to reason and understand the behavior of objects and their properties. It encompasses reasoning about physical concepts, such as the properties of objects reasoning (gravity, mass, inertia, or friction), their affordances, and how they can be manipulated.\n
Your task is to score the quality of the statement which can help answer a Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. Do not explain the reason. Only return an integer Score. 
Thought: {thought}
Score:
''',
    },
    "pubmedqa" : {
        "value_prompt_template":'''Score the quality of a thought generated to answer a Medical question.\n
Your task is to score the quality of the statement which can help answer a Question. Score should be an integer in the range 1(worst) - 10(best).\n

Analyze the following Question and score the Thought. The score should be an integer in the range 1-10. 
Question: {question}
Thought: {thought}

Do not give me your reasoning. Only return an integer Score. 
Score:
''',
    },
}

state_value = {
        "cose" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. 
Question: {question}
Thought: {thought}

It is very important that your Score is a single integer value. Do not give me your reasoning. Only return an integer Score. 
Score:
''',
    },
    "csqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}

It is very important that your Score is a single integer value. Do not give me your reasoning. Only return an integer Score. 
Score:
''',
    },
    "aqua" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
    "mathqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
    "medmcqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
    "medqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
    "piqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
    "pubmedqa" : {
        "value_prompt_template":'''Your task is to Score a Thought (between 1-10) which can help solve a Question. 1 being Worst and 10 being Best. It is very important that your Score is a single integer value.
Question: {question}
Thought: {thought}
Score:
''',
    },
}

#prompts for getting final answers
final_answers = {
            "cose" : {
        "final_answer_prompt_template":'''[INST]Answer a Commonsense Reasoning question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "csqa" : {
        "final_answer_prompt_template":'''[INST]Answer a Commonsense Reasoning question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "aqua" : {
        "final_answer_prompt_template":'''[INST]Answer an Algebra question. Algebra is a branch of Mathematics.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nPick one answer from these options: {answer_options}[\\INST]
''',
    },
    "mathqa" : {
        "final_answer_prompt_template":'''[INST]Answer a Mathematics question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "medmcqa" : {
        "final_answer_prompt_template":'''[INST]Answer a Medical question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "medqa" : {
        "final_answer_prompt_template":'''[INST]Answer a Medical question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "piqa" : {
        "final_answer_prompt_template":'''[INST]Answer a Physical Commonsense Reasoning question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
    "pubmedqa" : {
        "final_answer_prompt_template":'''[INST]The task of PubMedQA is to answer research questions with yes/no/maybe. Answer a Medical question.\n
The question is: {question}\n
To answer this question you have generated these thoughts:\n{thoughts}\n
ONLY GIVE AN EXACT RESPOND FROM THE OPTIONS YOU ARE GIVEN.\nAnswer from these options: {answer_options}[\\INST]
''',
    },
}