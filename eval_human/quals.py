'''distribute qualifications between workers'''

import json
#from collections import Counter
import xmltodict
import aws_config

from review_hits import get_meetup_tokens

WORK_TOKEN = {}
dial_pl = {}
W_IDS = []
NUM_OF_GAMES = {}

PATH_RESULTS = './results/'
LOGS_LOC = './logs/'

#QUAL = aws_config.ConnectToMTurk.mturk.create_qualification_type(
#    Name='MeetUpExp',
#    Description='monitoring the level of MeetUp awareness',
#    QualificationTypeStatus='Active'
#)
#LIST_QUAL = aws_config.ConnectToMTurk.mturk.list_qualification_types(
#    Query='MeetUpExp',
#    MustBeRequestable=True|False,
#)
#print(LIST_QUAL)

# qual typeid:
# 36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX

def assign_qual(data):
    '''assign qual value for the workers'''

    hits_data = json.load(data)
    for item in hits_data:
        assignment_list = aws_config.ConnectToMTurk.mturk.list_assignments_for_hit(
            HITId=item['hit_id'],
            AssignmentStatuses=['Approved', 'Rejected'],
            MaxResults=10
        )
        assignments = assignment_list['Assignments']
        for assignment in assignments:
            answer_dict = xmltodict.parse(assignment['Answer'])
            answer = answer_dict['QuestionFormAnswers']['Answer'][0]['FreeText']
            worker_id = assignment['WorkerId']
            
            # judge by the number of times a particular worker appears in the list
            # rather than by the workers' answers
            
            if answer is not None and (assignment['AssignmentStatus'] == 'Approved')\
            and not answer.startswith('-03') and len(answer) == 8:
                #print('APP', worker_id, answer)

                if worker_id not in WORK_TOKEN:
                    WORK_TOKEN[worker_id] = []
                WORK_TOKEN[worker_id].append(answer)
                W_IDS.append(worker_id)

                aws_config.ConnectToMTurk.mturk.associate_qualification_with_worker(
                    QualificationTypeId='36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX',
                    WorkerId=worker_id,
                    IntegerValue=1,
                    SendNotification=False
                )
                initial_qual_score = aws_config.ConnectToMTurk.mturk.get_qualification_score(
                    QualificationTypeId='36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX',
                    WorkerId=worker_id
                )
                print('qualification score for', worker_id, 'is',\
                 initial_qual_score['Qualification']['IntegerValue'])

            elif answer is not None and assignment['AssignmentStatus'] == 'Rejected':
                #print('REJ', worker_id, answer)
                aws_config.ConnectToMTurk.mturk.associate_qualification_with_worker(
                    QualificationTypeId='36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX',
                    WorkerId=worker_id,
                    IntegerValue=7,
                    SendNotification=False
                )
                initial_qual_score = aws_config.ConnectToMTurk.mturk.get_qualification_score(
                    QualificationTypeId='36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX',
                    WorkerId=worker_id
                )
                print('qualification score for', worker_id, 'is',\
                 initial_qual_score['Qualification']['IntegerValue'])
    print('------')

    for worker_id in W_IDS:
        count = W_IDS.count(worker_id)
        NUM_OF_GAMES[worker_id] = count

    for w_id in NUM_OF_GAMES:
        if NUM_OF_GAMES[w_id] > 1:

            #for dial_name in dial_and_tokens:
            #    if WORK_TOKEN[worker_id] == dial_and_tokens[dial_name][:-3]:
            #        print(dial_name, worker_id, count)
            #        dial_pl[worker_id] = dial_name.split('.log')[0].split('-')[5]

            aws_config.ConnectToMTurk.mturk.disassociate_qualification_from_worker(
                WorkerId=w_id,
                QualificationTypeId='3ETJLUMS0DM8X13DGYGLAJ6V7SNU3X'
            )
            aws_config.ConnectToMTurk.mturk.associate_qualification_with_worker(
                QualificationTypeId='3ETJLUMS0DM8X13DGYGLAJ6V7SNU3X',
                WorkerId=w_id,
                IntegerValue=NUM_OF_GAMES[w_id],
                SendNotification=False
            )
            updated_qual_score = aws_config.ConnectToMTurk.mturk.get_qualification_score(
                QualificationTypeId='3ETJLUMS0DM8X13DGYGLAJ6V7SNU3X',
                WorkerId=w_id
            )
            print('new qualification score is',\
             updated_qual_score['Qualification']['IntegerValue'], 'for', w_id)

            #if w_id in WORK_TOKEN:
            #    print(WORK_TOKEN, w_id, WORK_TOKEN[w_id])

    #print(dial_and_tokens)

    with open('./worker_ids.json', 'w+') as worker_ids_information:
        json.dump(NUM_OF_GAMES, worker_ids_information)

if __name__ == "__main__":
    with open('./results/results.json') as json_data:
        assign_qual(json_data)
