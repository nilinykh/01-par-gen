'''retreive results for HITs from AMT'''

import os
import glob
import json
import xmltodict
import aws_config

PATH_PUBLISHED = './published/'
PATH_RESULTS = './results/'

def get_results():
    '''ask AMT for results'''
    for filename in glob.glob(os.path.join(PATH_PUBLISHED, '*.json')):
        with open(filename, 'r') as handle:
            parsed = json.load(handle)
        for item in parsed:
            hit = aws_config.ConnectToMTurk.mturk.get_hit(HITId=item['hit_id'])
            item['status'] = hit['HIT']['HITStatus']
            assignments_list = aws_config.ConnectToMTurk.mturk.list_assignments_for_hit(
                HITId=item['hit_id'],
                AssignmentStatuses=['Submitted', 'Approved', 'Rejected'],
                MaxResults=10
            )
            
            assignments = assignments_list['Assignments']
            item['assignments_submitted_count'] = len(assignments)
            answers = []
            feedbacks = []
            for assignment in assignments:
                answer_dict = xmltodict.parse(assignment['Answer'])
                answer = answer_dict['QuestionFormAnswers']['Answer'][0]['FreeText']
                feedback = answer_dict['QuestionFormAnswers']['Answer'][1]['FreeText']
                answers.append(answer)
                feedbacks.append(feedback)

                # Approve the Assignment (if it hasn't been already)
                #if assignment['AssignmentStatus'] == 'Submitted':
                #    client.approve_assignment(
                #        AssignmentId=assignment_id,
                #        OverrideRejection=False
                #    )

            item['answers'] = answers
            item['feedback'] = feedbacks

            # rewrite here, check what can be changed / updated

            if not item['status'] == 'Disposed':
                with open(PATH_RESULTS+'results.json', 'a') as feedjson:
                    print(json.dumps(item, indent=2), file=feedjson)
                    print(',', file=feedjson)

if __name__ == "__main__":
    get_results()
    with open(PATH_RESULTS+'results.json', 'r+') as f:
        LINES = f.readlines()
        f.seek(0, 0)
        f.write('[')
        LINES[-1] = LINES[-1].replace(LINES[-1], ']')
        for line in LINES:
            f.write(line)
