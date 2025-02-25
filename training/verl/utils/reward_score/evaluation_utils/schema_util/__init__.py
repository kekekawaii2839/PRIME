import re
import jsonschema
from demjson3 import decode,JSONDecodeError,encode

import os
# if os.environ.get('NO_TOS', 'False') == 'True':
#     import json as json5
# else:
#     import json5
import json5

FINE_GRAINED_RESCALE = 0.3

def fgevaluate_with_schema(json_str: str, schema: dict) -> float:
    # find json, jsonc, json with comments
    try:    
        result = decode(json_str,return_errors=True)
                
        fine_percent = 1.0
        
        if result.errors:
            for e in result.errors:
                if isinstance(e,JSONDecodeError):
                    if "Comments are not allowed" in e.message:
                        continue
                    else:
                        # print("Error: ",e.message)
                        fine_percent = e.position.char_position/len(json_str)
                else:
                    raise e
                
    except:
        print("Failed To Decode:\n",json_str[:50])
        return 0

    try:
        jsonschema.validate(result.object,schema)
    except jsonschema.ValidationError as e:
        _o = result.object
        if len(e.absolute_path) == 0:
            return 0
        for p in list(e.absolute_path)[:-1]:
            _o = _o[p]
        SPECIAL_TAG =  "<|INVALID TAG FOR SCHEMA CHECK|>"
        _o[e.absolute_path[-1]] = SPECIAL_TAG
        # recalculate fine_percent
        encoded_o = encode(result.object,encoding="UTF-8",escape_unicode=False).decode("UTF-8")
        # print("Encoded: ",encoded_o)
        fine_percent *= max(encoded_o.find(SPECIAL_TAG)-1,0)/max(len(encoded_o)-len(SPECIAL_TAG),1)
        # print("Error: ",e.message)
    except:
        return 0

    if fine_percent < 1.0:
        fine_percent = fine_percent * FINE_GRAINED_RESCALE

    return fine_percent

def evaluate_tool(model_output: str, ground_truth: str) -> bool:
    try:
        ground_truth_call = json5.loads(ground_truth)
    except:
        ground_truth_call = None # irrelevant for tool calling
    try:
        # find json, jsonc, json with comments
        json_str = re.findall(r"```json\n(.*?)\n```", model_output, re.DOTALL)
        if json_str:
            model_call = json5.loads(json_str[0])
        else:
            model_call = json5.loads(model_output)
    except:
        return (ground_truth_call is None)
    # here is relevant for tool calling
    if ground_truth_call is None:
        return False # model should not call tools since ground truth does not call tools
    return ground_truth_call == model_call
    

def evaluate_schema(model_output: str, ground_truth: str, task: str, use_fine_grained: bool) -> bool|float:
    if task == "tool_calling":
        return evaluate_tool(model_output, ground_truth)

    ground_truth_schema = json5.loads(ground_truth)
    if task == "custom":
        ground_truth_schema = ground_truth_schema['verify_schema']
    elif task == "limitation":
        ground_truth_schema = ground_truth_schema['original']['schema']
    elif task == "translation":
        ground_truth_schema = ground_truth_schema['verify_schema']
    elif task == "schema":
        ground_truth_schema = ground_truth_schema
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
    # print('DEBUG in schema: success ground truth output using json5')
    
    try:
        # find json, jsonc, json with comments
        json_str = re.findall(r"```json\n(.*?)\n```", model_output, re.DOTALL)
        if json_str:
            if use_fine_grained:
                return fgevaluate_with_schema(json_str[0], ground_truth_schema)
            parsed_model_output = json5.loads(json_str[0])
        else:
            if use_fine_grained:
                return fgevaluate_with_schema(model_output, ground_truth_schema)
            parsed_model_output = json5.loads(model_output)
        print('DEBUG in schema: success parsing model output using regex')
    except:
        print("Failed To Parse:\n",model_output[:50])
        return False
    
    else:
        try:
            jsonschema.validate(parsed_model_output, ground_truth_schema)
            print('DEBUG in schema: success validating model output using jsonshema')
            return True
        except jsonschema.ValidationError as e:
            return False

    return False
