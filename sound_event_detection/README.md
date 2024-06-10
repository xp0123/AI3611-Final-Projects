数据说明：
* weak: 每个音频片段标注了有哪些事件发生
* unlabel_in_domain: 无标注，事件分布与 `weak` 的事件分布相似
* unlabel_out_of_domain: 无标注，事件分布与 `weak` 的事件分布无关

准备好环境后, 跑 baseline 步骤 (注意将环境初始化部分改成自己的设置):

1. 提取特征：
```bash
cd data;
sbatch prepare_data.sh /lustre/home/acct-stu/stu282/Data/sound_event_detection/raw_data
cd ..;
```

2. 训练、测试:
```bash
sbatch run.sh
```

注: evaluate.py 用于计算指标，预测结果 `prediction.csv` 写成这样的形式 (分隔符为 `\t`):
```
filename        event_label     onset   offset
Y09RRavdW3C0_30.000_40.000.wav  Speech  0.000   1.000
YIZ_zfkNcxRQ_61.000_71.000.wav  Blender 8.000   9.000
......
```
调用方法：
```bash
python evaluate.py --prediction prediction.csv \
                   --label data/eval/label.csv \
                   --output result.txt
```


