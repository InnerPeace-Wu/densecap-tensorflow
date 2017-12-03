# prapare data
pip install opencv-python
apt-get -y update && apt-get install -y libsm6 libxext6
pip install --upgrade pip
pip install -r requirements.txt
cd /valohai/inputs
tar -xvzf ./vg_data/visual_genome.tar.gz
mv ./valohai/inputs/visual_genome/ ./
mkdir ./images
unzip -xvzf image_1/images.zip -d ./images
unzip -xvzf image_2/images2.zip -d ./images
ls
ckpt_path='/valohai/inputs/resnet'
data_dir='/valohai/inputs/visual_genome'
cd /valohai/repository
cd lib
make
cd ..

tar -czvf /valohai/outputs/output.tar.gz ./output
