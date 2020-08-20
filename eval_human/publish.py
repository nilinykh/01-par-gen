'''publish batches of HITs on MTurk'''
import time
import json
import argparse
import sched

import aws_config
from hit_generator import gen_ip_pair

HTML = open('./survey.html', 'r').read()
QUESTION_XML = """
        <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
        <HTMLContent><![CDATA[{}]]></HTMLContent>
        <FrameHeight>650</FrameHeight>
        </HTMLQuestion>"""
QUESTION = QUESTION_XML.format(HTML)
Q_ATTR = {
    # Amount of assignments per HIT
    'MaxAssignments': 1,
    # How long the task is available on MTurk (1 day)
    'LifetimeInSeconds': 60*1440,
    # How much time Workers have in order to complete each task (20 minutes)
    'AssignmentDurationInSeconds': 60*20,
    # the HIT is automatically approved after this number of minutes (0.5 day)
    'AutoApprovalDelayInSeconds': 60*720,
    # The reward we offer Workers for each task
    'Reward': '0.10',
    'Title': 'Rate Image Descriptions [a short survey]',
    'Keywords': 'image, description, paragraph, evaluation',
    'Description': 'Given an image and its description, \
                    rate this description based on several questions.'}

def publish(input_args):
    '''publish HITs based on list of (img_link, paragraph) tuples'''
    image_paragraph_pairs = gen_ip_pair(int(input_args.HITs_number), str(input_args.type))
    RES = []
    for (img_link, paragraph, str(input_args.type)) in image_paragraph_pairs:
        res = create(img_link, paragraph, str(input_args.type))
        RES.append(res)
    return RES

def create(img_link, paragraph, model_type):
    '''defining HITs' template for MTurk'''
    
    RESULTS = []
    
    new_hit = aws_config.ConnectToMTurk.mturk.create_hit(
        **Q_ATTR,
        Question=QUESTION.replace('${Image}', img_link).\
        replace('${Paragraph}', paragraph).\
        replace('${Model}', model_type),
        QualificationRequirements=[
            {
                'QualificationTypeId' : '36GCPOW9EJEYRZ5B0H6PGIGL7W0ATX',
                'Comparator' : 'NotIn',
                'IntegerValues' :
                    [
                        11, 12, 13, 14, 15, 16
                    ],
                'ActionsGuarded' : 'PreviewAndAccept'
            },
            {
                'QualificationTypeId' : '00000000000000000071',
                'Comparator' : 'In',
                'LocaleValues' : [
                    {'Country':'GB'}, {'Country':'US'},
                    {'Country':'AU'}, {'Country':'CA'},
                    {'Country':'IE'}
                    ],
                'ActionsGuarded': 'PreviewAndAccept'
            },
            {
                'QualificationTypeId' : '00000000000000000040',
                'Comparator' : 'GreaterThanOrEqualTo',
                'IntegerValues' : [
                    2000
                    ],
                'ActionsGuarded': 'PreviewAndAccept'
            }
        ])

    RESULTS.append({
        'link': img_link,
        'model_type': model_type,
        'paragraph': paragraph,
        'hit_id': new_hit['HIT']['HITId']
    })

    print('A new HIT has been created. You can preview it here:')
    print('https://worker.mturk.com/mturk/preview?groupId=' + new_hit['HIT']['HITGroupId'])
    print('HITID = ' + new_hit['HIT']['HITId'] + ' (Use to Get Results)')
    
    return RESULTS

def run_data_collection(schedule_publisher):
    '''publish n HITs every 60 seconds'''
    print('Relax and wait while your HITs are being published...')
    parser_variables = argparse.ArgumentParser(description='publishing HITs')
    parser_variables.add_argument('-n', '--HITs_number',
                                  help='amount of HITs to be published',
                                  default='2')
    parser_variables.add_argument('-t', '--type',
                                  help='type of paragraphs to use',
                                  default='VISUAL')
    args = parser_variables.parse_args()
    res_all = publish(args)
    moment = time.strftime("%Y-%b-%d__%H_%M_%S", time.localtime())
    with open('./published/data_'+moment+'.json', 'w') as outfile:
        json.dump(res_all, outfile)
    SCHEDULER.enter(20, 1, run_data_collection, (schedule_publisher,))

if __name__ == "__main__":
    SCHEDULER = sched.scheduler(time.time, time.sleep)
    SCHEDULER.enter(3, 1, run_data_collection, (SCHEDULER,))
    SCHEDULER.run()
