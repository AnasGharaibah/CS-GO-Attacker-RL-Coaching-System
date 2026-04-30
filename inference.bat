@echo off
python inference.py --model rl\models\attacker_ppo_final --yolo cs2_yolo_model\weights\best.pt %*
pause
