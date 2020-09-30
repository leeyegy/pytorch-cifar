for epsilon  in 0.06275
do
	for step in 20
	do 
		for mode in ckpt.pth ckpt_last.pth 
		do
		python eval_MDAttack.py --num-steps $step --epsilon $epsilon --test_model_path ../checkpoint/amart_atrades_ResNet18/beta_5.0/$mode --md | tee atrades_md_$mode\_$epsilon.txt
	done
done
done
for epsilon  in 0.06275
do
	for step in 20
	do 
		for mode in ckpt.pth ckpt_last.pth 
		do
		python eval_MDAttack.py --num-steps $step --epsilon $epsilon --test_model_path ../checkpoint/amart_atrades_ResNet18/beta_5.0/$mode --mdmt | tee atrades_mdmt_$mode\_$epsilon.txt
	done
done
done
