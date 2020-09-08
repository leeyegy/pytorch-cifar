for net in ResNet18
do 
for loss in CE
do 
	for attack in PGD
	do 
		for eps in 0.03137
		do
    python shadow_adv_train.py --val_only  --resume   --net $net --loss $loss --attack_method $attack --epsilon $eps --source-model-path checkpoint/CE_ResNet18/ckpt.pth 
done
done
done
done
