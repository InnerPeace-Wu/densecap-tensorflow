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
cd /valohai/repository
cd lib
make
cd ..
bash ./tests/dencap_oa_test.sh {parameters}
tar -czvf /valohai/outputs/output.tar.gz ./output
