import subprocess
import os
from glob import glob
import argparse
import pathlib

# Only perform inference
# Name of the model file you want to use ( See paper for the different version )
# Type of dataloading
# Absolute path of the file containing the *.png
# Type of the images { png, jpg }
# Relative path of the Network weights to load
# Absolute path where you want to same the output, will create the folder
# Binary option to save the flow
def recursive_compute_flownet2(root_input, root_output, relative_path) :
    for f in glob(f'{root_input}/{relative_path}/*') :
        if os.path.isdir(f) and 'GroundTruth' not in f:
            relative_output_path = f'{relative_path}{os.path.basename(f)}'
            output_path = f'{root_output}/{relative_output_path}'
            print(f'Running in {f}, output in {output_path}')
            recursive_compute_flownet2(root_input, root_output, relative_output_path)
            pathlib.Path(output_path).mkdir(exist_ok=True)
            print(f)
            '''
            code = subprocess.call(['./main.py', '--inference',\
                            '--model', 'FlowNet2',\
                            '--inference_dataset', 'ImagesFromFolder',\
                            '--inference_dataset_root', f,\
                            '--inference_dataset_iext', 'jpg',\
                            '--resume', 'networks/FlowNet2_checkpoint.pth.tar',\
                            '--save_flow_root', output_path,\
                            '--save_flow'])
            print('code : ', code)
            '''
            print(f'./../flowizeti/flowizeti/__main__.py {output_path}/*.flo',\
                             '--mode', 'RGB','--outdir', output_path)
            subprocess.call([f'./../flowizeti/flowizeti/__main__.py {output_path}/*.flo --mode RGB --outdir {output_path}'], shell=True)
            subprocess.call([f'./../flowizeti/flowizeti/__main__.py {output_path}/*.flo --outdir {output_path} --mode ANGLE --statf {output_path}/minmax.txt'], shell=True)

if __name__ =='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-ri','--root_input', type=str)
    parser.add_argument('-ro','--root_output', type=str)
    args = parser.parse_args()
    pathlib.Path(args.root_output).mkdir(exist_ok=True)
    recursive_compute_flownet2(args.root_input, args.root_output, '')
