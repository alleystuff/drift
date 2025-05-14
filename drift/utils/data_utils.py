import re
import os
import ast
import json
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def get_data_df(data_path='data/thought_data/llama/'):
    thought_data_files = os.listdir(data_path)

    all_data_df = pd.DataFrame()
    for file in thought_data_files:
        file_name = data_path+file
        with open(file_name, 'r') as file:
            # print(file_name)
            thought_data = json.load(file)
        all_data_df = pd.concat([all_data_df, pd.DataFrame(thought_data)])

    # print(f"Total data samples: {len(all_data_df)}")
    # print(all_data_df[['task_name', 'question']].groupby(['task_name']).count())
    return all_data_df

def get_train_test_split(task_name='cose', data_path="data/thought_data/llama/", test_sample_size=0.2, test_random_state=40):
    """_summary_

    Args:
        task_name (str, optional): name of the current task. Defaults to 'cose'.
        data_path (str, optional): path where thought data is saved. Defaults to "data/thought_data/llama/".
        test_sample_size (float, optional): size of the test dataset. Defaults to 0.2.
        test_random_state (int, optional): random state to select the same test samples every time and select the remaining samples as train to avoid overlap. Defaults to 40.

    Returns:
        dataframes: training samples, test_samples
    """
    from data.data_collection import TaskDataset
    
    task_dataset = TaskDataset()
    thought_data_df = get_data_df(data_path=data_path)

    columns = ['model_id', 'task_name', 'all_thoughts', 'question_row_id', 'question', 'pruned_graph', "unpreferred_graph"]
    preference_data = thought_data_df[columns]

    preference_data['raw_pruned_graph'] = preference_data['pruned_graph']#.apply(lambda x: [])
    preference_data['pruned_graph'] = preference_data['pruned_graph'].apply(lambda x: "\n".join([t[1] for t in json.loads(x)]))
    preference_data['unpreferred_graph'] = preference_data['unpreferred_graph'].apply(lambda x: "\n".join([t[1] for t in json.loads(x)]))
    preference_data['question_row_id'] = preference_data['question_row_id'].astype(int)
    

    # print(f"Current Task Data | {task_name}")
    main_task_dataset = task_dataset.get_dataset(
        file_name=task_name
    ).rename(columns={"Unnamed: 0": "question_row_id"})    

    # join preference data with questions to get answer choices
    preference_data = preference_data[preference_data['task_name']==task_name]
    task_questions_and_thoughts = pd.merge(preference_data, main_task_dataset, on=['question_row_id'], how="left")

    # set test samples using seed to select the same test samples every time
    test_size = int(len(task_questions_and_thoughts)*test_sample_size)
    test_samples = task_questions_and_thoughts.sample(
        n=test_size, 
        random_state=test_random_state
    )
    train_samples = task_questions_and_thoughts.iloc[~task_questions_and_thoughts.index.isin(test_samples.index)]
    print(f"TaskName: {task_name} | Train Size: {len(train_samples)}| Test Size: {len(test_samples)}")
    return train_samples, test_samples

def get_preference_dataset(df):
    """_summary_

    Args:
        df (pandas dataframe): train or test dataframe containing samples

    Returns:
        dataset: huggingface Dataset
    """
    from datasets import Dataset
    columns = ['question_x', 'pruned_graph', "unpreferred_graph"]
    train_preference_data = df[columns]

    train_preference_data.rename(columns={
        "question_x": "prompt",
        "pruned_graph": "chosen",
        "unpreferred_graph": "rejected"
    }, inplace=True)

    # create dataset for training
    preference_dataset = Dataset.from_dict(train_preference_data.to_dict(orient='list'))
    return preference_dataset

def get_task_data_dict(row, task_name="csqa"):
    """_summary_

    Args:
        row (_type_): row of the task dataframe which contains Tot samples
        task_name (str, optional): name of the current task. Defaults to "csqa".

    Returns:
        data_dict: dictionary of the parsed sample including question, thoughts, answer options, answer etc.
    """
    if task_name=="csqa":
        # Replace 'array(...)' with list manually
        data_str = row['choices'].replace("array", "").replace(", dtype=object", "").replace("dtype=object", "")
        data_dict = ast.literal_eval(data_str)
        # Convert values to Python lists (if needed)
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict["label"] = list(data_dict["label"])
        
        # if the text choices are not in a list then put in a list
        data_dict["answer_options"] = list(data_dict["text"]) if len(data_dict["text"])>0 else data_dict["text"]
        
        # if the list is in another list then pull it out
        if len(data_dict['text'])==1:
            data_dict['text'] = data_dict['text'][0]
        
        #pull the final answer to compare and measure accuracy
        data_dict["final_answer"] = row["answerKey"]+ ": " + data_dict['text'][data_dict['label'].index(row["answerKey"])]
        
    if task_name=="cose":
        data_dict = {}
        data_dict['answer_options'] = re.findall(r"'(.*?)'", row['choices'])
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict['final_answer'] = row['answer']
        
    if task_name=="aqua":
        data_dict = {}
        # data_dict['answer_options'] = re.findall(r"'(.*?)'", row['options'])
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict["rationale"] = row["rationale"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        
        # process string to pull final answer from the answer options
        formatted_str = re.sub(r"'\s+'", "', '", row['options'])
        options = [i.split(')') for i in eval(formatted_str)]
        options = [{i[0] : i[1]} for i in options]
        data_dict['answer_options'] = options
        data_dict['final_answer'] =  [i for i in options if list(i.keys())[0]==row['correct']][0]
    
    if task_name=="mathqa":
        data_dict = {}
        try:
            data_dict['answer_options'] = [{i.split(')')[0].strip():i.split(')')[1].strip()} for i in row['options'].split(",")]
        except IndexError:
            data_dict['answer_options'] = row['options']
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict["rationale"] = row["Rationale"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict['final_answer'] = row['correct']
        
    if task_name=="medmcqa":
        data_dict = {}
        data_dict['answer_options'] = [row['opa'], row['opb'], row['opc'], row['opd']]
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict["rationale"] = row["exp"] #explanation for the correct answer
        data_dict["final_answer"] = data_dict['answer_options'][int(row["cop"])] #index on the correct option in the answer options
    
    if task_name=="medqa":
        data_dict = {}
        # Preprocess: Add commas between dictionaries
        formatted_str = re.sub(r"}\s+{", "}, {", row['options'])
        # Convert string to Python list of dictionaries
        formatted_list = ast.literal_eval(formatted_str)
        data_dict['answer_options'] = [{i['key']: i['value']} for i in formatted_list]
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict["final_answer"] = row["answer"]
    
    if task_name=="piqa":
        data_dict = {}
        data_dict['answer_options'] = [{0: row['sol1']}, {1: row['sol2']}]
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict['final_answer'] = list([i for i in data_dict['answer_options'] if list(i.keys())[0]==row['label']][0].values())[0]
    
    if task_name=="pubmedqa":
        data_dict = {}
        data_dict['answer_options'] = ['no', 'yes', 'maybe']
        data_dict["question"] = row['question_x']
        data_dict["thoughts"] = row["pruned_graph"]
        data_dict['thought_sentences'] = [t[1] for t in json.loads(row['raw_pruned_graph'])]
        data_dict['rationale'] = row['long_answer']
        data_dict['final_answer'] = row['final_decision']
        
    return data_dict