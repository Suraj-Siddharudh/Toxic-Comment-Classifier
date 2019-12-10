This file outlines the instructions to finetune BERT on Google Cloud Platform

Running BERT on GCP, training and prediction takes more than 8 hours 

1. Dowmload bert base uncased model - https://github.com/google-research/bert
2. Create project on GCP
3. Create Google Bucket 
'aldatoxiclabel' is the name of the bucket for all futher references. 
--upload the folder of bert base uncased model from step 1
--upload final_test.csv, final_train.csv, final_val.csv
3. Create virtual machine instance with gpu to the project

4. Run the following commands in the terminal in python3

sudo apt-get update
sudo apt-get install git-core
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py
sudo pip install numpy
sudo pip install sklearn
sudo pip install tensorflow-gpu==1.14
sudo apt-get install screen
git clone https://github.com/google-research/bert.git

5. Copy the files from bucket to instance using below commands 
** NOTE: change bucket name if different (aldatoxiclabel)

mkdir bert/data
mkdir bert/data/uncased_L-12_H-768_A-12
gsutil cp -p gs://aldatoxiclabel/toxic/final_test.csv bert/data
gsutil cp -p gs://aldatoxiclabel/toxic/final_train.csv bert/data
gsutil cp -p gs://aldatoxiclabel/toxic/final_val.csv bert/data
gsutil cp -p gs://aldatoxiclabel/uncased_L-12_H-768_A-12/bert_config.json bert/data/uncased_L-12_H-768_A-12
gsutil cp -p gs://aldatoxiclabel/uncased_L-12_H-768_A-12/vocab.txt bert/data/uncased_L-12_H-768_A-12
gsutil cp -p gs://aldatoxiclabel/uncased_L-12_H-768_A-12/bert_model.ckpt.meta bert/data/uncased_L-12_H-768_A-12
gsutil cp -p gs://aldatoxiclabel/uncased_L-12_H-768_A-12/bert_model.ckpt.index bert/data/uncased_L-12_H-768_A-12
gsutil cp -p gs://aldatoxiclabel/uncased_L-12_H-768_A-12/bert_model.ckpt.data-00000-of-00001 bert/data/uncased_L-12_H-768_A-12  

6. Upload run_classifier.py from submission folder into the terminal and then run
mv run_classifier.py bert

7. Run screen and run the following commands in the terminal

export BERT_BASE_DIR=data/uncased_L-12_H-768_A-12

python ./run_classifier.py \
--task_name=multi \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=data \
--vocab_file=${BERT_BASE_DIR}/vocab.txt \
--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=alda-output 

8. Ignore all the warnings that are printed

9. The results are stored in "test_results.csv"

10. Now we can use this file to run metrics