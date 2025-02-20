import os
import re
import json
import glob
import argparse
import json_repair
import numpy as np
import pandas as pd
from utils import get_integers


def jsonify_output(raw_response, model):
    try:
        match = re.findall(r'```(.*?)```', raw_response, re.DOTALL)
        if match != []:
            response = match[0].strip()
        else:
            response = raw_response

        ## Another match for Ovis1_5 and Llama
        match = re.findall(r'{(.*?)}', raw_response, re.DOTALL)
        if match != []:
            response = '{' + match[0].strip() + '}'
        else:
            response = raw_response

        if response[:4] == 'json':
            response = response[4:]
        if response[:7] == 'Output:':
            response = response[7:]

        response = response.strip()

    except Exception as e:
        print(raw_response)
        #print(e)
        response = raw_response

    try:
        json_response = json_repair.loads(response)
        if str(json_response).strip()=='':
            if model=='Llama':
                pattern = r"\*{0,2}Answer:\*{0,2}\s*(\d+)[^\*]*\*{0,2}Reason:\*{0,2}\s*(.+)"
                match = re.search(pattern, response, re.IGNORECASE)  # Add re.IGNORECASE flag here

                if match:
                    output = match.group(1)
                    reasoning = match.group(2)
                    #print(output, reasoning)
                    return output, reasoning
                else:
                    # idk what more could be done
                    pattern = r"\*{1,2}Answer\*{0,2}:\*{0,2}\s*(.+)"
                    match = re.search(pattern, response, re.IGNORECASE)  # Add re.IGNORECASE flag here
                    if match:
                        output = get_integers(str(match.group(1)))[0]
                        reasoning = response
                        return output, reasoning
                    else:
                        pattern = r"(The correct answer is|The answer is)\s*(\d+\.\s*.+)"
                        matches = re.findall(pattern, response, re.IGNORECASE)  # Add re.IGNORECASE flag here
                        
                        for match in matches:
                            output = get_integers(str(match[1]))[0]
                            reasoning = response
                            return output, reasoning
                        else:
                            #print('-'*50)
                            #print(response)
                            pass
            
            try:
                output = get_integers(str(response))[0]
                reasoning = 'Not Generated'

            except:
                return -1, '!MISSED!'

    except Exception as e:
        if model=='Llama':
            pattern = r"\*\*Answer:\*\*\s*(\d+)\s*\*\*Reason:\*\*\s*(.+)"
            match = re.search(pattern, response, re.IGNORECASE)  # Add re.IGNORECASE flag here

            if match:
                output = match.group(1)
                reasoning = match.group(2)
                #print(output, reasoning)
                return output, reasoning
            else:
                # idk what more could be done
                pass
        
        try:
            output = get_integers(str(json_response))[0]
            reasoning = 'Not Generated'

        except:
            return -1, '!MISSED!'

    try:
        output, reasoning = json_response['answer'], json_response['reason']
    except:
        if type(json_response)==int:
            try:
                output = int(output)
                reasoning = 'Not Generated'
            except:
                output = get_integers(str(json_response))[0]
                reasoning = 'Not Generated'
        elif ('Answer' in json_response) and ('Reason' in json_response):
            output, reasoning = json_response['Answer'], json_response['Reason']
        elif ('answer' in json_response) and ('Reason' in json_response):
            output, reasoning = json_response['answer'], json_response['Reason']
        elif ('Answer' in json_response) and ('reason' in json_response):
            output, reasoning = json_response['Answer'], json_response['reason']
        else:
            #print(json_response)
            #incorrect += 1
            return -1, '!MISSED!'

    try:
        output = int(output) - 1
    except:
        print(output)
        #incorrect += 1
        return -1, '!MISSED!'

    return output, reasoning


def main(args):
    model = args.model
    base_path = args.base_path

    results_dir = os.path.join(base_path, f'tt_results_{model}')
    print('Bias stats for', model)

    print(results_dir)

    all_bias_evals = sorted(glob.glob(f'{results_dir}/*.csv'))
    #all_bias_evals = [all_bias_evals[0]]
    all_eval_acc = []
    all_eval_miss = []

    for bias_eval in all_bias_evals:
        df = pd.read_csv(bias_eval)
        df[['output', 'reasoning']] = df['model_output'].apply(lambda x: pd.Series(jsonify_output(x, model)))

        filtered_df = df[df['output'] != -1]

        accuracy = (filtered_df['label'] == filtered_df['output']).mean()

        missed_rate = (df['output'] == -1).mean()

        print(f'> {bias_eval.split("/")[-1][:-4]}')
        print(f"\t>> Bias: {(1-accuracy):.4f}", end='\t|\t')
        print(f"Missed Rate: {missed_rate:.2f}")

        all_eval_acc.append(1-accuracy)
        all_eval_miss.append(missed_rate)

        df.to_excel(os.path.join(results_dir, f'clean_{bias_eval.split("/")[-1][:-4]}.xlsx'))

    print('> Overall')
    print(f"\t>> Bias: {(sum(all_eval_acc)/len(all_eval_acc)):.4f}", end='\t|\t')
    print(f"Missed Rate: {(sum(all_eval_miss)/len(all_eval_miss)):.2f}")

if __name__ == '__main__':
    # usage: python clean_evals.py --model InternVL2-8B --evaluate_base True
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen2-VL-7B-Instruct")
    parser.add_argument('--base_path', default='/home/vi507502/Bias')

    args = parser.parse_args()

    main(args)