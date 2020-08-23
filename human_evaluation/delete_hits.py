import sys
import argparse

import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[utils.get_parent_parser()],
              description="Delete HITs")
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--reset_hits', action='store_true', default=False)
    args = parser.parse_args()

    if (args.hit_ids_file is not None) == args.all:
        print('Must specify exactly one of --hit_ids_file or --all')
        sys.exit(1)

    mtc = utils.get_mturk_connection_from_args(args)

    if args.all:
        hit_ids = []
        for hit in mtc.get_all_hits():
            hit_ids.append(hit.HITId)
           
    elif args.reset_hits:
        hit_ids = []
        all_hits = mtc.list_hits(MaxResults=100)
        for i in all_hits['HITs']:
            hit_ids.append(i['HITId'])
            
    else:
        with open(args.hit_ids_file, 'r') as f:
            hit_ids = [line.strip() for line in f]
            
    print('This will delete %d HITs with sandbox=%s'
           % (len(hit_ids), str(args.sandbox)))
    print('Continue?')
    s = input('(Y/N): ')
    if s == 'Y' or s == 'y':
        for index, hit_id in enumerate(hit_ids):
            try:
                mtc.update_expiration_for_hit(HITId=hit_id, ExpireAt=0)
                mtc.delete_hit(HITId=hit_id)
                print('disabling: %d / %d' % (index+1, len(hit_ids)))
            except:
                print('Failed to delete: %s' % (hit_id))
    else:
        print('Aborting')
