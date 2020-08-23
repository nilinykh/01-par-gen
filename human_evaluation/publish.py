'''publish batches of HITs on MTurk'''
import json
import argparse
import sched, time
import os
import sys
from itertools import islice

import utils

s = sched.scheduler(time.time, time.sleep)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def publish(mtc, hit_properties, hit_ids_file, hit_chunk):
    
    for i, hit_input in enumerate(hit_chunk):

        template_params = {'input': json.dumps(hit_input)}
        html_doc = template.render(template_params)

        img_link, paragraph, model_type = hit_input

        html_doc = html_doc.replace('${Image}', img_link).\
            replace('${Paragraph}', paragraph).\
            replace('${Model}', model_type)

        html_question = '''
        <HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
          <HTMLContent>
            <![CDATA[
              <!DOCTYPE html>
              %s
            ]]>
          </HTMLContent>
          <FrameHeight>%d</FrameHeight>
        </HTMLQuestion>
        ''' % (html_doc, frame_height)

        hit_properties['Question'] = html_question
    
        launched = False
        while not launched:
            try:
                boto_hit = mtc.create_hit(**hit_properties)
                launched = True
            except Exception as e:
                print(e)
        hit_id = boto_hit['HIT']['HITId']
        hit_ids_file.write('%s\n' % hit_id)
        print('Launched HIT ID: %s, %d' % (hit_id, i + 1))


def create(args):

    with open(args.hit_ids_file, 'w') as hit_ids_file:
        
        input_hits = []
        for num, line in enumerate(args.input_json_file):
            hit_input = json.loads(line.strip())
            input_hits.append(hit_input)
        
        input_hits_chunks = chunk(input_hits, 10)
        
        for hit_chunk in input_hits_chunks:
            
            s.enter(60, 1, publish, (mtc, hit_properties, hit_ids_file, hit_chunk,))
            s.run()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(parents=[utils.get_parent_parser()])
    parser.add_argument('--hit_properties_file', type=argparse.FileType('r'))
    parser.add_argument('--html_template')
    parser.add_argument('--input_json_file', type=argparse.FileType('r'))
    args = parser.parse_args()

    # connect to MTurk
    mtc = utils.get_mturk_connection_from_args(args)  
    
    # load hit properties
    hit_properties = json.load(args.hit_properties_file)
    hit_properties['Reward'] = str(hit_properties['Reward'])
    
    # prepare qualifications
    utils.setup_qualifications(hit_properties, mtc)
    
    frame_height = hit_properties.pop('FrameHeight')
    env = utils.get_jinja_env(args.config)
    template = env.get_template(args.html_template)
    
    if args.hit_ids_file is None:
        print('Need to input a hit_ids_file')
        sys.exit()
    if os.path.isfile(args.hit_ids_file):
        print('hit_ids_file already exists')
        #sys.exit()
        
    create(args)