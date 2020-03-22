from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='vg',
                       paragraph_json_path='/home/xilini/par-gen/im2p_new/data/dataset_paragraphs_v2.json',
                       image_folder='/home/xilini/vis-data/',
                       densecap_feat_folder='/home/xilini/out/',
                       max_sentences=6,
                       min_word_freq=3,
                       output_folder='/home/xilini/par-gen/im2p_new/data/',
                       encoder_type='densecap',
                       max_words=50)
