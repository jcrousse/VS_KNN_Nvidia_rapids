import json

with open('config.json', 'r') as f:
    project_config = json.load(f)

SESSION_ID = project_config['df_columns']['session_id']
TIMESTAMP = project_config['df_columns']['timestamp']
ITEM_ID = project_config['df_columns']['item_id']
CATEGORY = project_config['df_columns']['category']
ITEM_POSITION = project_config['df_columns']['item_position']
PI_I =  project_config['df_columns']['pi_i']
