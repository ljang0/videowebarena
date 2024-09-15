import json

def check_errors(file_name):
    with open(file_name, 'r') as f:
        tasks = json.load(f)
    for i,task in enumerate(tasks):
        if task['task_id'] != i:
            print(f"{file_name}: Task {i} has an incorrect task_id")
            task['task_id'] = i
        intent_template = task['intent_template']
        instantiation = task['instantiation_dict']
        try:
            intent = intent_template.format(**instantiation)
        except (KeyError, IndexError) as e:
            print(f"{file_name}: Task {i} has an error in intent_template or instantiation_dict")
            print(e)
        
        if intent != task['intent']:
            if intent == task['intent'] + "." or intent == task['intent'] + "?":
                task['intent'] = intent
            else:
                print(f"{file_name}: Task {i} has an incorrect intent")
                print(f"intent: {task['intent']}")
                print(f"with instantiation and template: {intent}")
                print()
    
    with open(file_name, 'w') as f:
        json.dump(tasks, f, indent=4)

check_errors('test_maps.raw.json')

