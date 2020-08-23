import argparse, json

import utils
import xmltodict
import sys
import os
import re

def process_assignments(mtc, hit_id, status):
    results = []
    paginator = mtc.get_paginator('list_assignments_for_hit')
    try:
        for a_page in paginator.paginate(HITId=hit_id, PaginationConfig={'PageSize': 100}):            
            for a in a_page['Assignments']:
                
                extracted_res = {}

                if a['AssignmentStatus'] not in status:
                    continue
                    
                answers = xmltodict.parse(a['Answer'])['QuestionFormAnswers']['Answer']
                for each_a in answers:
                    if each_a['QuestionIdentifier'].startswith('wordChoice'):
                        extracted_res['wordChoice'] = int(each_a['FreeText'])
                    if each_a['QuestionIdentifier'].startswith('objSalience'):
                        extracted_res['objSalience'] = int(each_a['FreeText'])
                    if each_a['QuestionIdentifier'].startswith('sentStructure'):
                        extracted_res['sentStructure'] = int(each_a['FreeText'])
                    if each_a['QuestionIdentifier'].startswith('parCoherence'):
                        extracted_res['parCoherence'] = int(each_a['FreeText'])
                    if each_a['QuestionIdentifier'].startswith('feedback'):
                        extracted_res['feedback'] = str(each_a['FreeText'])

                results.append({
                  'assignment_id': a['AssignmentId'],
                  'hit_id': hit_id,
                  'worker_id': a['WorkerId'],
                  'output': extracted_res,
                  'submit_time': str(a['SubmitTime']),
              })
        
    except mtc.exceptions.RequestError:
        print('Bad hit_id %s' % str(hit_id), file=sys.stderr)
        return results

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[utils.get_parent_parser()])
    parser.add_argument('--output_file')
    parser.add_argument('--rejected', action='store_true', dest='rejected',
                      default=False,
                      help="Whether to also fetch the rejected hits.")
    args = parser.parse_args()
        
    mtc = utils.get_mturk_connection_from_args(args)

    results = []
    status = ['Approved', 'Submitted']
        
    if args.rejected:
        status = ['Approved', 'Submitted', 'Rejected']

    if args.hit_ids_file is None:
        for hit in mtc.get_all_hits():
            results += process_assignments(mtc, hit.HITId, status)
    # if the file exists, update it with the new results
    elif args.output_file != None and os.path.isfile(args.output_file):
        hit_dict = {}
        with open(args.output_file, 'r') as f1:
            output = json.load(f1)
            for item in output:
                hit_dict[item['hit_id']] = item
        for line in open(args.hit_ids_file, 'r'):
            hit_id = line.strip()
            if hit_id in hit_dict:
                results += [hit_dict[hit_id]]
            else:
                results += process_assignments(mtc, hit_id, status)
    else:
        with open(args.hit_ids_file, 'r') as f:
            for line in f:
                hit_id = line.strip()
                results += process_assignments(mtc, hit_id, status)

    with open(args.output_file, 'w') as out:
        json.dump(results, out)

