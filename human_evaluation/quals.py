'''distribute qualifications between workers'''

import json
import argparse, json
import os, sys

import utils



def assign_qual(args, qual_id, worker_info, hit_ids_checked):
    '''assign qual value for the workers'''
    
    paginator = mtc.get_paginator('list_assignments_for_hit')
        
    for line in open(args.hit_ids_file, 'r'):
        hit_id = line.strip()
        
        # only if this HIT has not been seen before
        if hit_id not in hit_ids_checked['ids_checked']:
                    
            for a_page in paginator.paginate(HITId=hit_id, PaginationConfig={'PageSize': 1}):

                # if there are any results for this HIT
                if a_page['NumResults'] != 0:

                    hit_id = a_page['Assignments'][0]['HITId']
                    worker_id = a_page['Assignments'][0]['WorkerId']
                    assignment_status = a_page['Assignments'][0]['AssignmentStatus']
                    
                    if assignment_status in ['Submitted', 'Approved', 'Rejected']:
                        hit_ids_checked['ids_checked'].append(hit_id)

                        if worker_id in worker_info.keys():
                            
                            if worker_info[worker_id] < 11:

                                worker_info[worker_id] += 1

                                if worker_info[worker_id] >= 11:

                                    mtc.associate_qualification_with_worker(
                                        QualificationTypeId=qual_id,
                                        WorkerId=worker_id,
                                        IntegerValue=11,
                                        SendNotification=False
                                    )

                                if worker_info[worker_id] < 11:

                                    mtc.associate_qualification_with_worker(
                                        QualificationTypeId=qual_id,
                                        WorkerId=worker_id,
                                        IntegerValue=worker_info[worker_id],
                                        SendNotification=False
                                    )

                            else:
                                worker_info[worker_id] += 1

                        if worker_id not in worker_info.keys():

                            worker_info[worker_id] = 1

                            mtc.associate_qualification_with_worker(
                                QualificationTypeId=qual_id,
                                WorkerId=worker_id,
                                IntegerValue=worker_info[worker_id],
                                SendNotification=False
                            )

        with open('./results/worker_file.json', 'w') as w2:
            json.dump(worker_info, w2)

        with open('./results/hit_ids_checked.json', 'w') as w3:
            json.dump(hit_ids_checked, w3)


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(parents=[utils.get_parent_parser()])
    args = parser.parse_args()

    # connect to MTurk
    mtc = utils.get_mturk_connection_from_args(args)
        
    if os.path.isfile('./results/worker_file.json'):
        with open('./results/worker_file.json', 'r') as f1:
            worker_info = json.load(f1)
    else:
        worker_info = {}
        with open('./results/worker_file.json', 'w') as f:
            json.dump({}, f)
            
    if os.path.isfile('./results/hit_ids_checked.json'):
        with open('./results/hit_ids_checked.json', 'r') as f3:
            hit_ids_checked = json.load(f3)
    else:
        hit_ids_checked = {}
        hit_ids_checked['ids_checked'] = []
        with open('./results/hit_ids_checked.json', 'w') as f4:
            json.dump(hit_ids_checked, f4)
         
    #mtc.delete_qualification_type(QualificationTypeId='3JC3H4XMU94HLX4F5GVWZBSKEQV3CH')
    
    #qual = mtc.create_qualification_type(
    #    Name='text_eval',
    #    Description='monitoring the level of text evaluation awareness',
    #    QualificationTypeStatus='Active'
    #)
    
    #mtc.disassociate_qualification_from_worker(
    #    QualificationTypeId='3918HVXFGHPANCD4BRXAV1HS97VXTT',
    #    WorkerId='A2H5A9GZYR4ELN'
    #)
    
    #mtc.associate_qualification_with_worker(
    #    QualificationTypeId='3918HVXFGHPANCD4BRXAV1HS97VXTT',
    #    WorkerId='A2H5A9GZYR4ELN',
    #    IntegerValue=10,
    #    SendNotification=False
    #)

    list_qual = mtc.list_qualification_types(
            Query='text_eval',
            MustBeRequestable=True|False,
        )
    
    # qual: 3918HVXFGHPANCD4BRXAV1HS97VXTT
        
    qual_id = list_qual['QualificationTypes'][0]['QualificationTypeId']
    
    assign_qual(args, qual_id, worker_info, hit_ids_checked)
    

