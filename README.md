# UNET_model
* Task : Segmentation
* Image Dataset : ISBI 2012 EM segmentation
  - Data Link : https://github.com/alexklibisz/isbi-2012
* Train code
'''python
"./drive/MyDrive/training_unet/train.py"
--lr 1e-2 --batch_size 2 --num_epoch 200
--data_dir "./drive/MyDrive/training_unet/dataset"
--ckpt_dir "./drive/MyDrive/training_unet/checkpoint_v2"
--log_dir "./drive/MyDrive/training_unet/log_v2"
--result_dir "./drive/MyDrive/training_unet/results_v2"
--mode "train"
--train_continue "off"
'''
* Test code
'''python
"./drive/MyDrive/training_unet/train.py"
--lr 1e-2 --batch_size 2 --num_epoch 200
--data_dir "./drive/MyDrive/training_unet/dataset"
--ckpt_dir "./drive/MyDrive/training_unet/checkpoint_v2"
--log_dir "./drive/MyDrive/training_unet/log_v2"
--result_dir "./drive/MyDrive/training_unet/results_v2"
--mode "test"
--train_continue "off"
'''