import argparse
import matplotlib.pyplot as plt
import os



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--content_image', type=str)
    parser.add_argument('--style_image', type=str)
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_DATA'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print('CAMILA!'*100)
    print(args)
    
    output_dir = '/opt/ml/output/data/'
    
    output_path = os.path.join(output_dir, args.content_image)
    print('output_path', output_path)
    plt.savefig(output_path)